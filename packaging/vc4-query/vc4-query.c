/*
 * vc4-query - the deviceQuery analogue for the VideoCore IV stack.
 *
 * Prints, in one command: the engine/daemon versions, the kernel, the V3D
 * revision and QPU count, the core and V3D clocks, die temperature, the
 * throttle register, CMA total/used, the daemon arena size, the op registry
 * counts, whether the installed engine can train, and the installed package
 * versions.
 *
 * It never opens /dev/mem.  Registers come from /sys/kernel/vc4mon (vc4top's
 * read-only module) or from the vc4d HELLO reply; clocks and the throttle
 * register come from vcgencmd, which talks to the firmware over /dev/vcio.
 *
 *   vc4-query [--json]
 *
 * Exit status: 0 on success (even if the daemon is down: the report says so),
 * 1 if the engine library could not be loaded.
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
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/utsname.h>
#include <unistd.h>

#include "vc4infer.h"
#include "vc4d_proto.h"

#define SOCK_PATH "/run/vc4d.sock"
#define SYSCACHE  "/usr/lib/vc4infer/kcache"

static int g_json;

/* ------------------------------------------------------------------ utils */
static int read_text(const char *path, char *buf, size_t n)
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

/* run a command, return its first line (no trailing newline) or "" */
static int run_line(const char *cmd, char *out, size_t n)
{
	FILE *f = popen(cmd, "r");
	if (!f)
		return -1;
	if (!fgets(out, (int)n, f)) {
		pclose(f);
		return -1;
	}
	pclose(f);
	out[strcspn(out, "\r\n")] = 0;
	return 0;
}

static uint32_t parse_hex_after(const char *s, const char *label)
{
	const char *p = strstr(s, label);
	if (!p)
		return 0;
	p += strlen(label);
	while (*p == ' ' || *p == ':' || *p == '\t')
		++p;
	return (uint32_t)strtoul(p, NULL, 16);
}

static int count_blobs(const char *dir)
{
	DIR *d = opendir(dir);
	if (!d)
		return -1;
	int n = 0;
	struct dirent *e;
	while ((e = readdir(d))) {
		size_t l = strlen(e->d_name);
		if (l > 4 && strcmp(e->d_name + l - 4, ".bin") == 0)
			++n;
	}
	closedir(d);
	return n;
}

/* ---------------------------------------------------------- daemon hello */
static int daemon_hello(uint32_t *proto, uint32_t *arena,
			uint32_t *ident0, uint32_t *ident1)
{
	int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
	if (fd < 0)
		return -1;
	struct sockaddr_un sa;
	memset(&sa, 0, sizeof(sa));
	sa.sun_family = AF_UNIX;
	snprintf(sa.sun_path, sizeof(sa.sun_path), "%s", SOCK_PATH);
	if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
		close(fd);
		return -1;
	}

	struct __attribute__((packed)) {
		struct vc4d_hdr h;
		struct vc4d_hello_req q;
	} req;
	memset(&req, 0, sizeof(req));
	req.h.magic = VC4D_MAGIC;
	req.h.len = sizeof(req);
	req.h.op = VC4D_OP_HELLO;
	req.h.version = VC4D_VERSION;
	req.q.version = VC4D_VERSION;

	if (send(fd, &req, sizeof(req), 0) != (ssize_t)sizeof(req)) {
		close(fd);
		return -1;
	}

	struct __attribute__((packed)) {
		struct vc4d_hdr h;
		struct vc4d_hello_rsp r;
	} rsp;
	memset(&rsp, 0, sizeof(rsp));
	ssize_t got = recv(fd, &rsp, sizeof(rsp), 0);
	close(fd);
	if (got < (ssize_t)(sizeof(rsp.h) + sizeof(rsp.r)) ||
	    rsp.h.magic != VC4D_MAGIC || rsp.h.status != VC4D_OK)
		return -1;

	*proto = rsp.r.version;
	*arena = rsp.r.arena_bytes;
	*ident0 = rsp.r.ident0;
	*ident1 = rsp.r.ident1;
	return 0;
}

/* ------------------------------------------------------------- vcgencmd */
static int vcgencmd_clock(const char *name, uint64_t *hz)
{
	char cmd[64], line[128];
	snprintf(cmd, sizeof(cmd), "vcgencmd measure_clock %s 2>/dev/null", name);
	if (run_line(cmd, line, sizeof(line)) != 0)
		return -1;
	char *eq = strchr(line, '=');
	if (!eq)
		return -1;
	*hz = strtoull(eq + 1, NULL, 10);
	return 0;
}

static int vcgencmd_throttled(uint32_t *bits)
{
	char line[128];
	if (run_line("vcgencmd get_throttled 2>/dev/null", line, sizeof(line)) != 0)
		return -1;
	char *eq = strchr(line, '=');
	if (!eq)
		return -1;
	*bits = (uint32_t)strtoul(eq + 1, NULL, 16);
	return 0;
}

static int read_temp_c(double *c)
{
	char buf[64];
	if (read_text("/sys/class/thermal/thermal_zone0/temp", buf, sizeof(buf)) == 0) {
		*c = atof(buf) / 1000.0;
		return 0;
	}
	char line[128];
	if (run_line("vcgencmd measure_temp 2>/dev/null", line, sizeof(line)) == 0) {
		char *eq = strchr(line, '=');
		if (eq) {
			*c = atof(eq + 1);
			return 0;
		}
	}
	return -1;
}

static void read_cma(unsigned long *total_kb, unsigned long *free_kb)
{
	*total_kb = *free_kb = 0;
	FILE *f = fopen("/proc/meminfo", "r");
	if (!f)
		return;
	char line[256];
	while (fgets(line, sizeof(line), f)) {
		unsigned long v;
		if (sscanf(line, "CmaTotal: %lu kB", &v) == 1)
			*total_kb = v;
		else if (sscanf(line, "CmaFree: %lu kB", &v) == 1)
			*free_kb = v;
	}
	fclose(f);
}

/* ------------------------------------------------------ package versions */
#define NPF 10
static const char *PF_NAMES[NPF] = {
	"vc4infer", "vc4d", "vc4infer-dev", "vc4-query", "vc4-smi",
	"vc4train", "vc4top", "vc4top-dkms", "vc4-toolkit", "vc4-toolkit-fast",
};
static char pf_ver[NPF][64];

static void read_packages(void)
{
	char cmd[512] = "dpkg-query -W -f='${Package} ${Version}\\n'";
	size_t o = strlen(cmd);
	for (int i = 0; i < NPF; ++i)
		o += snprintf(cmd + o, sizeof(cmd) - o, " %s", PF_NAMES[i]);
	snprintf(cmd + o, sizeof(cmd) - o, " 2>/dev/null");

	FILE *f = popen(cmd, "r");
	if (!f)
		return;
	char line[256];
	while (fgets(line, sizeof(line), f)) {
		char name[64], ver[64];
		if (sscanf(line, "%63s %63s", name, ver) != 2)
			continue;
		for (int i = 0; i < NPF; ++i)
			if (strcmp(PF_NAMES[i], name) == 0)
				snprintf(pf_ver[i], sizeof(pf_ver[i]), "%s", ver);
	}
	pclose(f);
}

/* ------------------------------------------------------------- registry */
typedef struct {
	const char *name;
	int supported;
} opinfo_t;

static opinfo_t *g_ops;
static size_t g_nops;

static int op_supported(const char *name)
{
	for (size_t i = 0; i < g_nops; ++i)
		if (strcmp(g_ops[i].name, name) == 0)
			return g_ops[i].supported;
	return -1;
}

static const char *FWD_OPS[] = {
	"matmul", "conv2d/3x3", "relu", "maxpool/2x2s2", "softmax",
};
static const char *BWD_OPS[] = {
	"matmul_bt", "transpose", "relu_bwd", "rsqrt", "softmax_ce_bwd",
	"im2col", "col2im", "conv2d/bwd_dx", "conv2d/bwd_dw", "maxpool_bwd",
};

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv)
{
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--json") == 0)
			g_json = 1;
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			printf("usage: vc4-query [--json]\n"
			       "  prints driver/daemon version, kernel, QPU count, clocks,\n"
			       "  temperature, throttled, CMA, arena, op counts, training\n"
			       "  support, cache paths and installed package versions.\n"
			       "  Reads /sys/kernel/vc4mon and vcgencmd; never opens /dev/mem.\n");
			return 0;
		} else {
			fprintf(stderr, "vc4-query: unknown argument %s\n", argv[i]);
			return 2;
		}
	}

	/* registry: the engine library is the only hard dependency */
	size_t cnt = 0;
	const vc4_op_info *t = vc4_supported_ops(&cnt);
	g_nops = cnt;
	g_ops = calloc(cnt ? cnt : 1, sizeof(*g_ops));
	if (!g_ops)
		return 1;
	int nsup = 0;
	for (size_t i = 0; i < cnt; ++i) {
		g_ops[i].name = t[i].name;
		g_ops[i].supported = t[i].supported;
		nsup += t[i].supported ? 1 : 0;
	}

	read_packages();

	/* daemon */
	uint32_t proto = 0, arena = 0, d_ident0 = 0, d_ident1 = 0;
	int have_daemon = (daemon_hello(&proto, &arena, &d_ident0, &d_ident1) == 0);

	/* V3D identity: vc4mon is readable without root; the daemon also knows */
	char identbuf[512] = "";
	uint32_t ident0 = d_ident0, ident1 = d_ident1;
	if (read_text("/sys/kernel/vc4mon/ident", identbuf, sizeof(identbuf)) == 0) {
		uint32_t v0 = parse_hex_after(identbuf, "IDENT0");
		uint32_t v1 = parse_hex_after(identbuf, "IDENT1");
		if (v0)
			ident0 = v0;
		if (v1)
			ident1 = v1;
	}
	int rev = ident1 & 0xF;
	int slices = (ident1 >> 4) & 0xF;
	int qps = (ident1 >> 8) & 0xF;
	int tmus = (ident1 >> 12) & 0xF;
	int nqpu = slices * qps;

	/* clocks, temperature, throttle, CMA */
	uint64_t core_hz = 0, v3d_hz = 0;
	int have_core = (vcgencmd_clock("core", &core_hz) == 0);
	int have_v3d = (vcgencmd_clock("v3d", &v3d_hz) == 0);
	double temp_c = 0;
	int have_temp = (read_temp_c(&temp_c) == 0);
	uint32_t thr = 0;
	int have_thr = (vcgencmd_throttled(&thr) == 0);
	unsigned long cma_total = 0, cma_free = 0;
	read_cma(&cma_total, &cma_free);

	struct utsname un;
	if (uname(&un) != 0) {
		snprintf(un.sysname, sizeof(un.sysname), "?");
		snprintf(un.release, sizeof(un.release), "?");
		snprintf(un.machine, sizeof(un.machine), "?");
	}

	/* training coverage */
	int nbwd = 0, nbwd_ok = 0;
	for (size_t i = 0; i < sizeof(BWD_OPS) / sizeof(BWD_OPS[0]); ++i) {
		++nbwd;
		nbwd_ok += (op_supported(BWD_OPS[i]) == 1);
	}
	int nfwd = 0, nfwd_ok = 0;
	for (size_t i = 0; i < sizeof(FWD_OPS) / sizeof(FWD_OPS[0]); ++i) {
		++nfwd;
		nfwd_ok += (op_supported(FWD_OPS[i]) == 1);
	}

	int sysblobs = count_blobs(SYSCACHE);
	char sysblob_str[64];
	if (sysblobs < 0)
		snprintf(sysblob_str, sizeof(sysblob_str), "not installed");
	else
		snprintf(sysblob_str, sizeof(sysblob_str), "%d blobs", sysblobs);

	/* ------------------------------------------------------------ json */
	if (g_json) {
		printf("{\n");
		printf("  \"tool\": \"vc4-query\",\n");
		printf("  \"engine\": \"%s\",\n", VC4INFER_VERSION_STRING);
		printf("  \"kernel\": \"%s\",\n", un.release);
		printf("  \"machine\": \"%s\",\n", un.machine);
		printf("  \"v3d\": {\"ident0\": %u, \"ident1\": %u, \"revision\": %d, "
		       "\"slices\": %d, \"qpus_per_slice\": %d, \"qpus\": %d, "
		       "\"tmus\": %d},\n",
		       ident0, ident1, rev, slices, qps, nqpu, tmus);
		printf("  \"backend\": \"%s\",\n",
		       have_daemon ? "vc4d" : (geteuid() == 0 ? "direct" : "none"));
		printf("  \"daemon\": {\"reachable\": %s, \"protocol\": %u, "
		       "\"arena_bytes\": %u},\n",
		       have_daemon ? "true" : "false", proto, arena);
		printf("  \"clocks_hz\": {\"core\": %llu, \"v3d\": %llu},\n",
		       (unsigned long long)core_hz, (unsigned long long)v3d_hz);
		printf("  \"temperature_c\": %.1f,\n", have_temp ? temp_c : -1.0);
		printf("  \"throttled\": %u,\n", thr);
		printf("  \"cma_kb\": {\"total\": %lu, \"free\": %lu},\n",
		       cma_total, cma_free);
		printf("  \"ops\": {\"total\": %zu, \"supported\": %d, "
		       "\"unsupported\": %d},\n", cnt, nsup, (int)cnt - nsup);
		printf("  \"training\": {\"supported\": %s, \"backward_ops\": %d, "
		       "\"backward_ops_supported\": %d, \"forward_with_backward\": %d, "
		       "\"forward_ops_supported\": %d},\n",
		       (nbwd_ok == nbwd && nbwd > 0) ? "true" : "false",
		       nbwd, nbwd_ok, nfwd, nfwd_ok);
		printf("  \"kernel_cache\": {\"system_dir\": \"%s\", "
		       "\"system_blobs\": ", SYSCACHE);
		if (sysblobs < 0)
			printf("null},\n");
		else
			printf("%d},\n", sysblobs);
		printf("  \"packages\": {");
		int first = 1;
		for (int i = 0; i < NPF; ++i) {
			if (!pf_ver[i][0])
				continue;
			printf("%s\"%s\": \"%s\"", first ? "" : ", ", PF_NAMES[i],
			       pf_ver[i]);
			first = 0;
		}
		printf("}\n}\n");
		return 0;
	}

	/* ------------------------------------------------------------ text */
	char pkgline[2048] = "";
	for (int i = 0; i < NPF; ++i) {
		if (!pf_ver[i][0])
			continue;
		size_t o = strlen(pkgline);
		snprintf(pkgline + o, sizeof(pkgline) - o, "%s%s %s",
			 o ? ", " : "", PF_NAMES[i], pf_ver[i]);
	}

	printf("vc4-query 1.0.0 -- VideoCore IV device query\n");
	printf("--------------------------------------------------------------\n");
	printf("engine:          libvc4infer.so.1 (API %s)\n",
	       VC4INFER_VERSION_STRING);
	printf("daemon:          ");
	if (have_daemon)
		printf("vc4d %s, protocol %u, arena %u bytes (%.1f MiB)\n",
		       pf_ver[1][0] ? pf_ver[1] : "(unpackaged copy)",
		       proto, arena, (double)arena / (1 << 20));
	else
		printf("not reachable (%s)", SOCK_PATH);
	if (!have_daemon)
		printf("\n");
	printf("backend:         %s\n",
	       have_daemon ? "vc4d (/run/vc4d.sock)"
			   : (geteuid() == 0 ? "direct /dev/mem (root)"
					     : "none (start vc4d, or run as root)"));
	printf("kernel:          %s (%s)\n", un.release, un.machine);
	printf("soc:             VideoCore IV (V3D) rev %d, %d slices x %d QPUs "
	       "= %d QPUs, %d TMUs\n", rev, slices, qps, nqpu, slices * tmus);
	printf("clocks:          ");
	if (have_core)
		printf("core %.0f MHz", (double)core_hz / 1e6);
	else
		printf("core ?");
	if (have_v3d)
		printf(", V3D %.0f MHz", (double)v3d_hz / 1e6);
	else
		printf(", V3D ?");
	printf("  (stock; this stack never changes them)\n");
	printf("temperature:     ");
	if (have_temp)
		printf("%.1f C\n", temp_c);
	else
		printf("unavailable\n");
	if (have_thr)
		printf("throttled:       0x%x (%s)\n", thr,
		       thr ? ((thr & 0xF) ? "ACTIVE" : "historical bits only")
			   : "none");
	else
		printf("throttled:       unavailable\n");
	printf("cma:             %lu MiB total, %lu MiB free (%lu MiB used)\n",
	       cma_total / 1024, cma_free / 1024,
	       (cma_total - cma_free) / 1024);
	printf("arena bytes:     %u\n", arena);
	printf("ops:             %zu registry entries, %d dispatchable on the QPU, "
	       "%zu not implemented\n", cnt, nsup, cnt - (size_t)nsup);
	printf("training:        %s -- %d/%d backward/training ops dispatchable, "
	       "%d/%d forward ops with a backward\n",
	       (nbwd_ok == nbwd && nbwd > 0) ? "yes" : "NO", nbwd_ok, nbwd,
	       nfwd_ok, nfwd);
	printf("system kcache:   %s (%s)\n", SYSCACHE, sysblob_str);
	printf("packages:        %s\n", pkgline[0] ? pkgline : "(none found)");
	return 0;
}
