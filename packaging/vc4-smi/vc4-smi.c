/*
 * vc4-smi - the nvidia-smi analogue for the VideoCore IV stack.
 *
 * One-shot text summary: state, QPU utilisation, stock clocks, temperature,
 * throttle bits, CMA memory, QPU state and the clients currently attached to
 * vc4d.  It is a thin wrapper over `vc4top --once` (NO_COLOR=1) plus the
 * read-only /sys/kernel/vc4mon register window; it never opens /dev/mem and
 * never writes a register.
 *
 *   vc4-smi [--json]
 *
 * Exit status: 0 healthy, 1 unhealthy (active throttle bits, undervoltage, a
 * scheduler queue error, or V3D ERRSTAT error bits other than VCD-idle),
 * 2 usage/vc4top missing.
 */
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define SOCK_PATH "/run/vc4d.sock"
#define REGS_PATH "/sys/kernel/vc4mon/regs"
#define ERRSTAT_OFF 0x00f20u
#define ERRSTAT_VCD_IDLE 0x00001000u

static int g_json;

/* ------------------------------------------------------------------ utils */
static int grab(const char *buf, const char *label, char *out, size_t n)
{
	const char *p = strstr(buf, label);
	if (!p)
		return -1;
	p += strlen(label);
	while (*p == ' ' || *p == '\t')
		++p;
	const char *e = strchr(p, '\n');
	size_t l = e ? (size_t)(e - p) : strlen(p);
	if (l >= n)
		l = n - 1;
	memcpy(out, p, l);
	out[l] = 0;
	while (l > 0 && (out[l - 1] == ' ' || out[l - 1] == '\r'))
		out[--l] = 0;
	return 0;
}

/* "400000000 Hz (400.000 MHz)" -> "400.000 MHz" */
static void fmt_mhz(const char *raw, char *out, size_t n)
{
	const char *o = strchr(raw, '(');
	if (o) {
		const char *c = strchr(o, ')');
		if (c) {
			size_t l = (size_t)(c - o - 1);
			if (l >= n)
				l = n - 1;
			memcpy(out, o + 1, l);
			out[l] = 0;
			return;
		}
	}
	unsigned long long hz = strtoull(raw, NULL, 10);
	snprintf(out, n, "%.0f MHz", (double)hz / 1e6);
}

/* "200 / 256 MiB (78.1% used, ...)" -> "200/256 MiB" */
static void fmt_cma(const char *raw, char *out, size_t n)
{
	unsigned long used = 0, total = 0;
	if (sscanf(raw, "%lu / %lu MiB", &used, &total) == 2)
		snprintf(out, n, "%lu/%lu MiB", used, total);
	else
		snprintf(out, n, "%s", raw);
}

static int read_all(const char *path, char *buf, size_t n)
{
	int fd = open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return -1;
	ssize_t got = read(fd, buf, n - 1);
	close(fd);
	if (got < 0)
		return -1;
	buf[got] = 0;
	return 0;
}

/* all of vc4top --once, plain text */
static char *run_once(void)
{
	FILE *f = popen("NO_COLOR=1 vc4top --once 2>/dev/null", "r");
	if (!f)
		return NULL;
	size_t cap = 65536, len = 0;
	char *buf = malloc(cap);
	if (!buf) {
		pclose(f);
		return NULL;
	}
	size_t got;
	while ((got = fread(buf + len, 1, cap - len - 1, f)) > 0) {
		len += got;
		if (len + 1 >= cap)
			break;
	}
	pclose(f);
	buf[len] = 0;
	return buf;
}

/* read one V3D register from the vc4mon dump; -1 when unavailable */
static int read_reg(uint32_t off, uint32_t *val)
{
	char buf[16384];
	if (read_all(REGS_PATH, buf, sizeof(buf)) != 0)
		return -1;
	FILE *f = fmemopen(buf, strlen(buf), "r");
	if (!f)
		return -1;
	unsigned o, v;
	int found = -1;
	while (fscanf(f, "%x %x", &o, &v) == 2) {
		if (o == off) {
			*val = v;
			found = 0;
			break;
		}
	}
	fclose(f);
	return found;
}

/* --------------------------------------------------------- client scan */
static ino_t sock_inode(void)
{
	struct stat st;
	if (stat(SOCK_PATH, &st) != 0)
		return 0;
	return st.st_ino;
}

static int list_clients(char *out, size_t n)
{
	ino_t ino = sock_inode();
	if (!ino) {
		snprintf(out, n, "none (vc4d not running)");
		return 0;
	}
	DIR *p = opendir("/proc");
	if (!p) {
		snprintf(out, n, "unknown");
		return 0;
	}
	int count = 0;
	size_t o = 0;
	struct dirent *de;
	while ((de = readdir(p))) {
		if (!isdigit((unsigned char)de->d_name[0]))
			continue;
		char fdd[64];
		snprintf(fdd, sizeof(fdd), "/proc/%s/fd", de->d_name);
		DIR *fd = opendir(fdd);
		if (!fd)
			continue;
		struct dirent *fe;
		int hit = 0;
		while ((fe = readdir(fd))) {
			char link[400], target[256];
			snprintf(link, sizeof(link), "%s/%s", fdd, fe->d_name);
			ssize_t l = readlink(link, target, sizeof(target) - 1);
			if (l <= 0)
				continue;
			target[l] = 0;
			if (strncmp(target, "socket:[", 8) == 0 &&
			    (ino_t)strtoull(target + 8, NULL, 10) == ino) {
				hit = 1;
				break;
			}
		}
		closedir(fd);
		if (!hit)
			continue;
		char comm[64] = "?";
		char cp[320];
		snprintf(cp, sizeof(cp), "/proc/%s/comm", de->d_name);
		int cf = open(cp, O_RDONLY | O_CLOEXEC);
		if (cf >= 0) {
			ssize_t l = read(cf, comm, sizeof(comm) - 1);
			close(cf);
			if (l > 0) {
				comm[l] = 0;
				comm[strcspn(comm, "\n")] = 0;
			}
		}
		int w = snprintf(out + o, n - o, "%s%s(%s)", o ? ", " : "",
				 de->d_name, comm);
		if (w < 0 || (size_t)w >= n - o)
			break;
		o += (size_t)w;
		++count;
	}
	closedir(p);
	if (!count)
		snprintf(out, n, "none");
	return count;
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv)
{
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--json") == 0)
			g_json = 1;
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			printf("usage: vc4-smi [--json]\n"
			       "  one-shot summary: state, util, clocks, temp, throttle,\n"
			       "  CMA memory, QPU state and vc4d clients.\n"
			       "  Exits 1 if the GPU is unhealthy.\n");
			return 0;
		} else {
			fprintf(stderr, "vc4-smi: unknown argument %s\n", argv[i]);
			return 2;
		}
	}

	char *top = run_once();
	if (!top) {
		fprintf(stderr, "vc4-smi: vc4top is required (apt install vc4top)\n");
		return 2;
	}

	char temp[64] = "?", core_raw[96] = "", v3d_raw[96] = "";
	char thr_raw[64] = "?", cma_raw[96] = "?", err_raw[96] = "";
	char qps[96] = "?", slices[64] = "?", uvolt[64] = "?", srqcs[128] = "";
	char core[64] = "?", v3d[64] = "?", cma[64] = "?";
	grab(top, "temp:", temp, sizeof(temp));
	grab(top, "core clock:", core_raw, sizeof(core_raw));
	grab(top, "v3d clock:", v3d_raw, sizeof(v3d_raw));
	grab(top, "throttled:", thr_raw, sizeof(thr_raw));
	grab(top, "gpu cma used:", cma_raw, sizeof(cma_raw));
	grab(top, "ERRSTAT:", err_raw, sizeof(err_raw));
	grab(top, "QPUs/slice:", qps, sizeof(qps));
	grab(top, "slices:", slices, sizeof(slices));
	grab(top, "undervoltage:", uvolt, sizeof(uvolt));
	grab(top, "SRQCS:", srqcs, sizeof(srqcs));
	fmt_mhz(core_raw, core, sizeof(core));
	fmt_mhz(v3d_raw, v3d, sizeof(v3d));
	fmt_cma(cma_raw, cma, sizeof(cma));

	uint32_t thr = 0;
	{
		const char *p = strstr(thr_raw, "0x");
		if (p)
			thr = (uint32_t)strtoul(p, NULL, 16);
	}

	int qerr = 0;
	{
		const char *p = strstr(srqcs, "qerr=");
		if (p)
			qerr = atoi(p + 5);
	}

	uint32_t errstat = 0;
	int have_err = (read_reg(ERRSTAT_OFF, &errstat) == 0);
	if (!have_err) {
		const char *p = strstr(err_raw, "0x");
		if (p) {
			errstat = (uint32_t)strtoul(p, NULL, 16);
			have_err = 1;
		}
	}

	/* utilisation: sample the VCD-idle bit; a 200 ms window is long enough
	 * for every kernel this stack launches (a 256^3 matmul is ~4 ms) */
	int busy = 0, samples = 0;
	if (have_err) {
		for (int i = 0; i < 10; ++i) {
			uint32_t v;
			if (read_reg(ERRSTAT_OFF, &v) == 0) {
				++samples;
				if (!(v & ERRSTAT_VCD_IDLE))
					++busy;
			}
			usleep(20000);
		}
	}
	int util = samples ? (busy * 100 + samples / 2) / samples : -1;

	int undervolt = (uvolt[0] == '1');
	int err_active = have_err && (errstat & ~ERRSTAT_VCD_IDLE);
	int bad = (thr & 0xF) || undervolt || qerr != 0 || err_active;

	char clients[2048];
	int nclients = list_clients(clients, sizeof(clients));

	char when[64];
	time_t now = time(NULL);
	struct tm tmv;
	localtime_r(&now, &tmv);
	strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S %Z", &tmv);

	/* QPU text: "4 QPUs/slice (total 12 QPUs)" -> "12 QPUs / 3 slices" */
	int total_qpu = 0;
	{
		const char *p = strstr(qps, "total ");
		if (p)
			total_qpu = atoi(p + 6);
	}
	int nsl = atoi(slices);

	if (g_json) {
		printf("{\"state\": \"%s\", \"util_pct\": %d, \"core\": \"%s\", "
		       "\"v3d\": \"%s\", \"temp\": \"%s\", \"throttled\": %u, "
		       "\"cma\": \"%s\", \"qpus\": %d, \"slices\": %d, "
		       "\"errstat\": %u, \"qerr\": %d, \"clients\": %d}\n",
		       bad ? "alarm" : "ok", util, core, v3d, temp, thr, cma,
		       total_qpu, nsl, errstat, qerr, nclients);
		free(top);
		return bad ? 1 : 0;
	}

	printf("vc4-smi " VC4_SMI_VERSION "  --  VideoCore IV (V3D)  %s\n", when);
	printf("GPU  %d QPUs / %d slices   backend %s\n",
	       total_qpu, nsl, access(SOCK_PATH, F_OK) == 0 ? "vc4d" : "none");
	printf("\n");

	char util_str[16];
	if (util < 0)
		snprintf(util_str, sizeof(util_str), "?");
	else
		snprintf(util_str, sizeof(util_str), "%d%%", util);

	printf("%-7s %-6s %-12s %-12s %-8s %-10s %-12s %s\n",
	       "STATE", "UTIL", "CORE", "V3D", "TEMP", "THROTTLE", "CMA USED",
	       "QPU");
	printf("%-7s %-6s %-12s %-12s %-8s %-10s %-12s %s\n",
	       bad ? "ALARM" : "ok", util_str, core, v3d, temp, thr_raw, cma,
	       have_err ? ((errstat & ERRSTAT_VCD_IDLE) ? "idle" : "busy") : "?");
	printf("\n");
	printf("util:        %s\n",
	       util < 0 ? "unavailable (no /sys/kernel/vc4mon)"
			: (util == 0 ? "0% (idle over a 200 ms window)"
				     : "busy over a 200 ms window"));
	printf("errstat:    0x%08x%s\n", errstat,
	       err_active ? "  <-- ERROR BITS SET" : "");
	printf("undervoltage: %s\n", uvolt);
	printf("clients:    %s\n", clients);
	if (thr & 0xF)
		printf("ALARM: active throttle bits 0x%x -- the board is limiting "
		       "itself (stock clocks, no overclock involved)\n", thr & 0xF);
	if (undervolt)
		printf("ALARM: undervoltage reported by the firmware\n");
	if (qerr)
		printf("ALARM: scheduler queue error (%d)\n", qerr);
	if (err_active)
		printf("ALARM: V3D ERRSTAT 0x%08x has bits other than VCD-idle\n",
		       errstat);
	if (!bad)
		printf("\nGPU is healthy: no throttle, no undervoltage, no queue or "
		       "ERRSTAT errors.\n");

	free(top);
	return bad ? 1 : 0;
}
