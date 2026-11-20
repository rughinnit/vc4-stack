/*
 * libpjrt_vc4.so - a PJRT C API plugin for the VideoCore IV QPU engine.
 *
 * The plugin is loaded by jaxlib (GetPjrtApi) and turns JAX's StableHLO
 * module into libvc4infer calls.  The ABI is the XLA PJRT C API from
 * third_party/pjrt_c_api.h; see docs/PJRT.md for the provenance of the
 * header and for the supported op subset.
 *
 * Memory model: a PJRT buffer owns a 64-byte aligned host allocation.  The
 * contents of that allocation are the device value; Execute uploads the
 * inputs into the engine's dma-buf arena, runs the QPU kernels, and
 * downloads the results.  Transfers are therefore a memcpy, and no value
 * ever changes on the ARM between two kernels.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pjrt_c_api.h"
#include "vc4infer.h"
#include "stablehlo.h"

#define VC4PJRT_PLATFORM_NAME "vc4"
#define VC4PJRT_PLATFORM_VERSION "vc4infer 1.2"
#define VC4PJRT_DEVICE_KIND "VideoCore IV QPU"
#define VC4PJRT_DEVICE_DEBUG "vc4:0 (BCM2837 V3D, 400 MHz)"

/* ------------------------------------------------------------------ logging */

static FILE *log_file(void) {
    static FILE *f;
    static int initialised;
    if (!initialised) {
        const char *p = getenv("VC4PJRT_LOG");
        initialised = 1;
        if (p && *p) f = fopen(p, "a");
    }
    return f;
}

static void vlogf(const char *fmt, ...) {
    FILE *f = log_file();
    if (!f) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fflush(f);
}

/* -------------------------------------------------------------- error objects */

typedef struct {
    PJRT_Error base;
    PJRT_Error_Code code;
    char *message;
} VC4Error;

static void vc4_error_destroy(PJRT_Error *error);
static void vc4_error_message(const PJRT_Error *error, const char **message,
                              size_t *message_size);
static PJRT_Error_Code vc4_error_code(const PJRT_Error *error);
static void vc4_error_payloads(const PJRT_Error *error,
                               PJRT_Error_PayloadVisitor visitor, void *arg);

static const PJRT_Error_FunctionTable g_error_vtable = {
    sizeof(PJRT_Error_FunctionTable),
    sizeof(VC4Error),
    NULL,
    vc4_error_destroy,
    vc4_error_message,
    vc4_error_code,
    vc4_error_payloads,
};

static PJRT_Error *make_error(PJRT_Error_Code code, const char *fmt, ...) {
    VC4Error *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->base.vtable = &g_error_vtable;
    e->code = code;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    e->message = strdup(buf);
    vlogf("error: %s\n", buf);
    return &e->base;
}

static void vc4_error_destroy(PJRT_Error *error) {
    if (!error) return;
    VC4Error *e = (VC4Error *)error;
    free(e->message);
    free(e);
}

static void vc4_error_message(const PJRT_Error *error, const char **message,
                              size_t *message_size) {
    const VC4Error *e = (const VC4Error *)error;
    *message = e && e->message ? e->message : "";
    *message_size = e && e->message ? strlen(e->message) : 0;
}

static PJRT_Error_Code vc4_error_code(const PJRT_Error *error) {
    return ((const VC4Error *)error)->code;
}

static void vc4_error_payloads(const PJRT_Error *error,
                               PJRT_Error_PayloadVisitor visitor, void *arg) {
    (void)error;
    (void)visitor;
    (void)arg;
}

/* --------------------------------------------------------------- event objects */

struct PJRT_Event {
    bool ready;
    PJRT_Error_Code code;
    char *message;
    PJRT_Event_OnReadyCallback callback;
    void *user_arg;
};

static PJRT_Event *event_new_ready(PJRT_Error_Code code, const char *message) {
    PJRT_Event *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->ready = true;
    e->code = code;
    e->message = message ? strdup(message) : NULL;
    return e;
}

static void event_fire_callback(PJRT_Event *e) {
    if (!e || !e->ready || !e->callback) return;
    PJRT_Event_OnReadyCallback cb = e->callback;
    void *arg = e->user_arg;
    e->callback = NULL;
    PJRT_Error *err = NULL;
    if (e->code != PJRT_Error_Code_OK)
        err = make_error(e->code, "%s", e->message ? e->message : "event error");
    cb(err, arg);
}

/* ------------------------------------------------------------------- objects */

typedef struct VC4PjrtClient VC4PjrtClient;

struct PJRT_DeviceDescription {
    int id;
    int process_index;
    int local_hardware_id;
    const char *kind;
    const char *debug_string;
    const char *to_string;
};

struct PJRT_Device {
    VC4PjrtClient *client;
    PJRT_DeviceDescription *description;
};

struct PJRT_Buffer {
    VC4PjrtClient *client;
    void *data;
    size_t nbytes;
    int64_t *dims;
    size_t num_dims;
    PJRT_Buffer_Type type;
    bool deleted;
    bool on_cpu; /* always false: the value lives in device memory */
};

struct PJRT_Executable {
    int refcount;
    char *code;        /* the StableHLO module, kept for error messages */
    size_t code_size;
    char *name;
    size_t num_outputs;
    PJRT_Buffer_Type *output_types;
    int64_t **output_dims;
    size_t *output_ranks;
    size_t num_parameters;
    vc4sl_program prog;     /* lowered instruction list */
    int have_prog;
    size_t *offs;           /* per tensor slot, byte offset in the slab */
    vc4_buf slab;
    int scratch_slot;
    size_t total_bytes;
};

struct PJRT_LoadedExecutable {
    PJRT_Executable *executable;
    PJRT_Device *device;
    bool deleted;
};

struct PJRT_TopologyDescription {
    int refcount;
};

struct VC4PjrtClient {
    PJRT_Device device;
    PJRT_DeviceDescription description;
    PJRT_Memory memory;
    PJRT_Device *device_list[1];
    PJRT_Memory *memory_list[1];
    PJRT_TopologyDescription *topology;
    struct PJRT_TopologyDescription topology_storage;
    vc4_ctx *ctx;
    int engine_tried;
    vc4_status engine_status;
};

static vc4_ctx *client_engine(VC4PjrtClient *c) {
    if (!c) return NULL;
    if (!c->engine_tried) {
        const char *cache = getenv("VC4INFER_CACHE");
        /* The installed engine defaults to the direct root path; on this
           board the GPU is owned by vc4d, so select the daemon backend unless
           the caller asked for something else. */
        if (!getenv("VC4INFER_DAEMON") && access("/run/vc4d.sock", F_OK) == 0)
            setenv("VC4INFER_DAEMON", "1", 0);
        c->engine_tried = 1;
        c->engine_status = vc4_init(&c->ctx, cache, 0);
        if (c->engine_status != VC4_OK) {
            vlogf("vc4_init failed: %s\n", vc4_strerror(c->engine_status));
            c->ctx = NULL;
        } else {
            vlogf("vc4_init ok: arena %zu B, %zu B used\n",
                  vc4_arena_bytes(c->ctx), vc4_arena_used(c->ctx));
        }
    }
    return c->ctx;
}

static size_t type_size(PJRT_Buffer_Type t) {
    switch (t) {
    case PJRT_Buffer_Type_PRED: return 1;
    case PJRT_Buffer_Type_S8:
    case PJRT_Buffer_Type_U8:
    case PJRT_Buffer_Type_F8E5M2:
    case PJRT_Buffer_Type_F8E4M3FN:
    case PJRT_Buffer_Type_F8E4M3B11FNUZ:
    case PJRT_Buffer_Type_F8E5M2FNUZ:
    case PJRT_Buffer_Type_F8E4M3FNUZ:
    case PJRT_Buffer_Type_F8E4M3:
    case PJRT_Buffer_Type_F8E3M4:
    case PJRT_Buffer_Type_F8E8M0FNU:
    case PJRT_Buffer_Type_S1:
    case PJRT_Buffer_Type_U1: return 1;
    case PJRT_Buffer_Type_S16:
    case PJRT_Buffer_Type_U16:
    case PJRT_Buffer_Type_F16:
    case PJRT_Buffer_Type_BF16: return 2;
    case PJRT_Buffer_Type_S32:
    case PJRT_Buffer_Type_U32:
    case PJRT_Buffer_Type_F32: return 4;
    case PJRT_Buffer_Type_S64:
    case PJRT_Buffer_Type_U64:
    case PJRT_Buffer_Type_F64:
    case PJRT_Buffer_Type_C64: return 8;
    case PJRT_Buffer_Type_C128: return 16;
    default: return 0;
    }
}

static size_t shape_bytes(PJRT_Buffer_Type type, const int64_t *dims,
                          size_t num_dims) {
    size_t es = type_size(type);
    if (!es) return 0;
    size_t n = es;
    for (size_t i = 0; i < num_dims; i++) {
        if (dims[i] < 0) return 0;
        n *= (size_t)dims[i];
    }
    return n;
}

static PJRT_Buffer *buffer_new(VC4PjrtClient *client, PJRT_Buffer_Type type,
                               const int64_t *dims, size_t num_dims) {
    PJRT_Buffer *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->client = client;
    b->type = type;
    b->num_dims = num_dims;
    if (num_dims) {
        b->dims = malloc(num_dims * sizeof(int64_t));
        if (!b->dims) { free(b); return NULL; }
        memcpy(b->dims, dims, num_dims * sizeof(int64_t));
    }
    b->nbytes = shape_bytes(type, dims, num_dims);
    if (!b->nbytes) b->nbytes = 1;
    if (posix_memalign(&b->data, 64, b->nbytes) != 0) {
        free(b->dims);
        free(b);
        return NULL;
    }
    memset(b->data, 0, b->nbytes);
    return b;
}

static void buffer_free(PJRT_Buffer *b) {
    if (!b) return;
    free(b->data);
    free(b->dims);
    free(b);
}

static PJRT_Executable *executable_new(void) {
    PJRT_Executable *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->refcount = 1;
    e->name = strdup("vc4_executable");
    return e;
}

static void executable_release(PJRT_Executable *e) {
    if (!e) return;
    if (--e->refcount > 0) return;
    free(e->code);
    free(e->name);
    if (e->have_prog) vc4sl_free(&e->prog);
    free(e->offs);
    for (size_t i = 0; i < e->num_outputs; i++) free(e->output_dims[i]);
    free(e->output_types);
    free(e->output_dims);
    free(e->output_ranks);
    free(e);
}

/* ---------------------------------------------------------- client / device */

static PJRT_DeviceDescription g_description = {
    0, 0, 0,
    VC4PJRT_DEVICE_KIND,
    VC4PJRT_DEVICE_DEBUG,
    "Vc4Device(id=0)",
};

static const PJRT_Memory_FunctionTable g_memory_vtable = {
    sizeof(PJRT_Memory_FunctionTable),
    NULL,
    sizeof(PJRT_Memory),
    NULL,
    NULL,
};

/* ------------------------------------------------------------- handler: misc */

static void h_Error_Destroy(PJRT_Error_Destroy_Args *args) {
    vc4_error_destroy(args->error);
}

static void h_Error_Message(PJRT_Error_Message_Args *args) {
    vc4_error_message(args->error, &args->message, &args->message_size);
}

static PJRT_Error *h_Error_GetCode(PJRT_Error_GetCode_Args *args) {
    args->code = vc4_error_code(args->error);
    return NULL;
}

static PJRT_Error *h_Error_ForEachPayload(PJRT_Error_ForEachPayload_Args *args) {
    (void)args;
    return NULL;
}

static PJRT_Error *h_Plugin_Initialize(PJRT_Plugin_Initialize_Args *args) {
    (void)args;
    vlogf("plugin initialise (PJRT API %d.%d)\n", PJRT_API_MAJOR,
          PJRT_API_MINOR);
    return NULL;
}

static PJRT_Error *h_Plugin_Attributes(PJRT_Plugin_Attributes_Args *args) {
    args->attributes = NULL;
    args->num_attributes = 0;
    return NULL;
}

/* ----------------------------------------------------------------- events */

static PJRT_Error *h_Event_Destroy(PJRT_Event_Destroy_Args *args) {
    if (args->event) {
        free(args->event->message);
        free(args->event);
    }
    return NULL;
}

static PJRT_Error *h_Event_IsReady(PJRT_Event_IsReady_Args *args) {
    args->is_ready = args->event ? args->event->ready : true;
    return NULL;
}

static PJRT_Error *h_Event_Error(PJRT_Event_Error_Args *args) {
    PJRT_Event *e = args->event;
    if (!e || e->code == PJRT_Error_Code_OK) return NULL;
    return make_error(e->code, "%s", e->message ? e->message : "event error");
}

static PJRT_Error *h_Event_Await(PJRT_Event_Await_Args *args) {
    PJRT_Event *e = args->event;
    if (!e || e->code == PJRT_Error_Code_OK) return NULL;
    return make_error(e->code, "%s", e->message ? e->message : "event error");
}

static PJRT_Error *h_Event_OnReady(PJRT_Event_OnReady_Args *args) {
    PJRT_Event *e = args->event;
    if (!e) return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "null event");
    e->callback = args->callback;
    e->user_arg = args->user_arg;
    event_fire_callback(e);
    return NULL;
}

static PJRT_Error *h_Event_Create(PJRT_Event_Create_Args *args) {
    args->event = calloc(1, sizeof(PJRT_Event));
    if (!args->event)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "event alloc");
    return NULL;
}

static PJRT_Error *h_Event_Set(PJRT_Event_Set_Args *args) {
    PJRT_Event *e = args->event;
    if (!e) return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "null event");
    e->ready = true;
    e->code = args->error_code;
    free(e->message);
    e->message = NULL;
    if (args->error_message && args->error_message_size)
        e->message = strndup(args->error_message, args->error_message_size);
    event_fire_callback(e);
    return NULL;
}

/* ----------------------------------------------------------------- client */

static PJRT_Error *h_Client_Create(PJRT_Client_Create_Args *args) {
    VC4PjrtClient *c = calloc(1, sizeof(*c));
    if (!c)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "client alloc");
    c->description = g_description;
    c->device.client = c;
    c->device.description = &c->description;
    c->memory.vtable = &g_memory_vtable;
    c->memory.local_device_id = 0;
    c->memory.memory_kind_id = 0;
    c->device_list[0] = &c->device;
    c->memory_list[0] = &c->memory;
    c->topology = &c->topology_storage;
    /* The client owns this topology; a PJRT_TopologyDescription_Destroy on the
       borrowed pointer must not free memory inside the client. */
    c->topology_storage.refcount = 0x7fffffff;
    args->client = (PJRT_Client *)c;
    vlogf("client created (%zu create options)\n", args->num_options);
    return NULL;
}

static PJRT_Error *h_Client_Destroy(PJRT_Client_Destroy_Args *args) {
    VC4PjrtClient *c = (VC4PjrtClient *)args->client;
    if (!c) return NULL;
    if (c->ctx) vc4_shutdown(c->ctx);
    free(c);
    return NULL;
}

static PJRT_Error *h_Client_PlatformName(PJRT_Client_PlatformName_Args *args) {
    args->platform_name = VC4PJRT_PLATFORM_NAME;
    args->platform_name_size = strlen(VC4PJRT_PLATFORM_NAME);
    return NULL;
}

static PJRT_Error *h_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args *args) {
    args->process_index = 0;
    return NULL;
}

static PJRT_Error *h_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args *args) {
    args->platform_version = VC4PJRT_PLATFORM_VERSION;
    args->platform_version_size = strlen(VC4PJRT_PLATFORM_VERSION);
    return NULL;
}

static PJRT_Error *h_Client_Devices(PJRT_Client_Devices_Args *args) {
    VC4PjrtClient *c = (VC4PjrtClient *)args->client;
    args->devices = c->device_list;
    args->num_devices = 1;
    vlogf("Client_Devices -> %p (%p)\n", (void *)c->device_list[0],
          (void *)c->device_list);
    return NULL;
}

static PJRT_Error *h_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args *args) {
    VC4PjrtClient *c = (VC4PjrtClient *)args->client;
    args->addressable_devices = c->device_list;
    args->num_addressable_devices = 1;
    vlogf("Client_AddressableDevices -> %p (%p)\n",
          (void *)c->device_list[0], (void *)c->device_list);
    return NULL;
}

static PJRT_Error *h_Client_LookupDevice(PJRT_Client_LookupDevice_Args *args) {
    VC4PjrtClient *c = (VC4PjrtClient *)args->client;
    if (args->id != 0)
        return make_error(PJRT_Error_Code_NOT_FOUND, "no vc4 device %d",
                          args->id);
    args->device = &c->device;
    return NULL;
}

static PJRT_Error *h_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args *args) {
    VC4PjrtClient *c = (VC4PjrtClient *)args->client;
    if (args->local_hardware_id != 0)
        return make_error(PJRT_Error_Code_NOT_FOUND,
                          "no vc4 addressable device %d",
                          args->local_hardware_id);
    args->addressable_device = &c->device;
    return NULL;
}

static PJRT_Error *h_Client_AddressableMemories(
    PJRT_Client_AddressableMemories_Args *args) {
    VC4PjrtClient *c = (VC4PjrtClient *)args->client;
    args->addressable_memories = c->memory_list;
    args->num_addressable_memories = 1;
    return NULL;
}

static PJRT_Error *h_Client_DefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args *args) {
    if (args->default_assignment_size <
        (size_t)(args->num_replicas * args->num_partitions))
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT,
                          "device assignment too small");
    for (int i = 0; i < args->num_replicas * args->num_partitions; i++)
        args->default_assignment[i] = 0;
    return NULL;
}

static PJRT_Error *h_Client_TopologyDescription(
    PJRT_Client_TopologyDescription_Args *args) {
    VC4PjrtClient *c = (VC4PjrtClient *)args->client;
    args->topology = (PJRT_TopologyDescription *)c->topology;
    return NULL;
}

/* --------------------------------------------------------- device and memory */

static PJRT_Error *h_Device_GetDescription(
    PJRT_Device_GetDescription_Args *args) {
    args->device_description = args->device->description;
    return NULL;
}

static PJRT_Error *h_Device_IsAddressable(
    PJRT_Device_IsAddressable_Args *args) {
    args->is_addressable = true;
    return NULL;
}

static PJRT_Error *h_Device_LocalHardwareId(
    PJRT_Device_LocalHardwareId_Args *args) {
    args->local_hardware_id = 0;
    return NULL;
}

static PJRT_Error *h_Device_AddressableMemories(
    PJRT_Device_AddressableMemories_Args *args) {
    VC4PjrtClient *c = args->device->client;
    args->memories = c->memory_list;
    args->num_memories = 1;
    return NULL;
}

static PJRT_Error *h_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args *args) {
    args->memory = &args->device->client->memory;
    return NULL;
}

static PJRT_Error *h_Device_MemoryStats(PJRT_Device_MemoryStats_Args *args) {
    args->bytes_in_use = 0;
    args->peak_bytes_in_use = 0;
    args->peak_bytes_in_use_is_set = true;
    args->num_allocs = 0;
    args->num_allocs_is_set = true;
    args->largest_alloc_size = 0;
    args->largest_alloc_size_is_set = true;
    args->bytes_limit = 0;
    args->bytes_limit_is_set = false;
    return NULL;
}

static PJRT_Error *h_Description_Id(PJRT_DeviceDescription_Id_Args *args) {
    args->id = args->device_description->id;
    return NULL;
}

static PJRT_Error *h_Description_ProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args *args) {
    args->process_index = args->device_description->process_index;
    return NULL;
}

static PJRT_Error *h_Description_Attributes(
    PJRT_DeviceDescription_Attributes_Args *args) {
    args->attributes = NULL;
    args->num_attributes = 0;
    return NULL;
}

static PJRT_Error *h_Description_Kind(PJRT_DeviceDescription_Kind_Args *args) {
    args->device_kind = args->device_description->kind;
    args->device_kind_size = strlen(args->device_description->kind);
    return NULL;
}

static PJRT_Error *h_Description_DebugString(
    PJRT_DeviceDescription_DebugString_Args *args) {
    args->debug_string = args->device_description->debug_string;
    args->debug_string_size = strlen(args->device_description->debug_string);
    return NULL;
}

static PJRT_Error *h_Description_ToString(
    PJRT_DeviceDescription_ToString_Args *args) {
    args->to_string = args->device_description->to_string;
    args->to_string_size = strlen(args->device_description->to_string);
    return NULL;
}

static PJRT_Error *h_Memory_Id(PJRT_Memory_Id_Args *args) {
    args->id = 0;
    return NULL;
}

static PJRT_Error *h_Memory_Kind(PJRT_Memory_Kind_Args *args) {
    args->kind = "device";
    args->kind_size = strlen("device");
    return NULL;
}

static PJRT_Error *h_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args *args) {
    args->kind_id = 0;
    return NULL;
}

static PJRT_Error *h_Memory_DebugString(PJRT_Memory_DebugString_Args *args) {
    args->debug_string = "Vc4Memory(id=0, device)";
    args->debug_string_size = strlen(args->debug_string);
    return NULL;
}

static PJRT_Error *h_Memory_ToString(PJRT_Memory_ToString_Args *args) {
    args->to_string = "Vc4Memory(id=0)";
    args->to_string_size = strlen(args->to_string);
    return NULL;
}

static PJRT_Error *h_Memory_AddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args *args) {
    VC4PjrtClient *c =
        (VC4PjrtClient *)((char *)args->memory -
                          offsetof(VC4PjrtClient, memory));
    args->devices = c->device_list;
    args->num_devices = 1;
    return NULL;
}

/* --------------------------------------------------------------- topology */

static PJRT_Error *h_Topology_Create(PJRT_TopologyDescription_Create_Args *args) {
    PJRT_TopologyDescription *t = calloc(1, sizeof(*t));
    if (!t)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "topology alloc");
    t->refcount = 1;
    args->topology = t;
    return NULL;
}

static PJRT_Error *h_Topology_Destroy(
    PJRT_TopologyDescription_Destroy_Args *args) {
    if (args->topology && --args->topology->refcount <= 0)
        free(args->topology);
    return NULL;
}

static PJRT_Error *h_Topology_PlatformName(
    PJRT_TopologyDescription_PlatformName_Args *args) {
    args->platform_name = VC4PJRT_PLATFORM_NAME;
    args->platform_name_size = strlen(VC4PJRT_PLATFORM_NAME);
    return NULL;
}

static PJRT_Error *h_Topology_PlatformVersion(
    PJRT_TopologyDescription_PlatformVersion_Args *args) {
    args->platform_version = VC4PJRT_PLATFORM_VERSION;
    args->platform_version_size = strlen(VC4PJRT_PLATFORM_VERSION);
    return NULL;
}

static PJRT_Error *h_Topology_GetDeviceDescriptions(
    PJRT_TopologyDescription_GetDeviceDescriptions_Args *args) {
    static PJRT_DeviceDescription *descriptions[1] = {&g_description};
    args->descriptions = descriptions;
    args->num_descriptions = 1;
    return NULL;
}

static void topology_bytes_deleter(PJRT_SerializedTopology *t) {
    free(t);
}

static PJRT_Error *h_Topology_Serialize(
    PJRT_TopologyDescription_Serialize_Args *args) {
    static const char payload[] = "vc4:1";
    char *copy = strdup(payload);
    if (!copy)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "topology copy");
    args->serialized_bytes = copy;
    args->serialized_bytes_size = strlen(payload);
    args->serialized_topology = (PJRT_SerializedTopology *)copy;
    args->serialized_topology_deleter = topology_bytes_deleter;
    return NULL;
}

static PJRT_Error *h_Topology_Deserialize(
    PJRT_TopologyDescription_Deserialize_Args *args) {
    PJRT_TopologyDescription *t = calloc(1, sizeof(*t));
    if (!t)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "topology alloc");
    t->refcount = 1;
    args->topology = t;
    return NULL;
}

static PJRT_Error *h_Topology_Attributes(
    PJRT_TopologyDescription_Attributes_Args *args) {
    args->attributes = NULL;
    args->num_attributes = 0;
    return NULL;
}

static PJRT_Error *h_Topology_Fingerprint(
    PJRT_TopologyDescription_Fingerprint_Args *args) {
    args->fingerprint = 0x7663340000000001ull;
    return NULL;
}

/* ----------------------------------------------------------------- buffers */

static bool layout_is_dense(const int64_t *dims, size_t num_dims) {
    (void)dims;
    (void)num_dims;
    return true;
}

static PJRT_Error *h_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args *args) {
    VC4PjrtClient *c = (VC4PjrtClient *)args->client;
    size_t es = type_size(args->type);
    if (!es)
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT,
                          "host transfer: unsupported element type %d",
                          (int)args->type);
    PJRT_Buffer *b = buffer_new(c, args->type, args->dims, args->num_dims);
    if (!b)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "buffer alloc");
    if (args->byte_strides && args->num_byte_strides) {
        size_t total = 1;
        for (size_t i = 0; i < args->num_dims; i++) total *= (size_t)args->dims[i];
        if (layout_is_dense(args->dims, args->num_dims) &&
            args->num_byte_strides == args->num_dims) {
            /* element-by-element in C order; strides may be negative */
            size_t *idx = calloc(args->num_dims ? args->num_dims : 1, sizeof(size_t));
            if (!idx) { buffer_free(b); return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "index alloc"); }
            const char *src = args->data;
            for (size_t n = 0; n < total; n++) {
                int64_t off = 0;
                for (size_t d = 0; d < args->num_dims; d++)
                    off += (int64_t)idx[d] * args->byte_strides[d];
                memcpy((char *)b->data + n * es, src + off, es);
                for (size_t d = args->num_dims; d-- > 0;) {
                    if (++idx[d] < (size_t)args->dims[d]) break;
                    idx[d] = 0;
                }
            }
            free(idx);
        } else {
            memset(b->data, 0, b->nbytes);
        }
    } else if (args->data) {
        memcpy(b->data, args->data, b->nbytes);
    }
    args->buffer = b;
    args->done_with_host_buffer = event_new_ready(PJRT_Error_Code_OK, NULL);
    return NULL;
}

static PJRT_Error *h_Client_CreateUninitializedBuffer(
    PJRT_Client_CreateUninitializedBuffer_Args *args) {
    PJRT_Buffer *b = buffer_new((VC4PjrtClient *)args->client,
                                args->shape_element_type, args->shape_dims,
                                args->shape_num_dims);
    if (!b)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "buffer alloc");
    args->buffer = b;
    return NULL;
}

static PJRT_Error *h_Buffer_Destroy(PJRT_Buffer_Destroy_Args *args) {
    buffer_free(args->buffer);
    return NULL;
}

static PJRT_Error *h_Buffer_ElementType(PJRT_Buffer_ElementType_Args *args) {
    args->type = args->buffer->type;
    return NULL;
}

static PJRT_Error *h_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args *args) {
    args->dims = args->buffer->dims;
    args->num_dims = args->buffer->num_dims;
    return NULL;
}

static PJRT_Error *h_Buffer_UnpaddedDimensions(
    PJRT_Buffer_UnpaddedDimensions_Args *args) {
    args->unpadded_dims = args->buffer->dims;
    args->num_dims = args->buffer->num_dims;
    return NULL;
}

static PJRT_Error *h_Buffer_DynamicDimensionIndices(
    PJRT_Buffer_DynamicDimensionIndices_Args *args) {
    args->dynamic_dim_indices = NULL;
    args->num_dynamic_dims = 0;
    return NULL;
}

static PJRT_Error *h_Buffer_GetMemoryLayout(
    PJRT_Buffer_GetMemoryLayout_Args *args) {
    PJRT_Buffer *b = args->buffer;
    static int64_t strides[8];
    if (b->num_dims > 8)
        return make_error(PJRT_Error_Code_UNIMPLEMENTED,
                          "buffer rank %zu exceeds the plugin's layout limit",
                          b->num_dims);
    /* major-to-minor dims; minor_to_major lists fastest-varying first. */
    size_t es = type_size(b->type);
    int64_t step = (int64_t)es;
    for (size_t i = b->num_dims; i-- > 0;) {
        strides[i] = step;
        step *= b->dims[i];
    }
    args->layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
    args->layout.extension_start = NULL;
    args->layout.type = PJRT_Buffer_MemoryLayout_Type_Strides;
    args->layout.strides.struct_size =
        PJRT_Buffer_MemoryLayout_Strides_STRUCT_SIZE;
    args->layout.strides.extension_start = NULL;
    args->layout.strides.byte_strides = strides;
    args->layout.strides.num_byte_strides = b->num_dims;
    return NULL;
}

static PJRT_Error *h_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args *args) {
    args->on_device_size_in_bytes = args->buffer->nbytes;
    return NULL;
}

static PJRT_Error *h_Buffer_Device(PJRT_Buffer_Device_Args *args) {
    args->device = &args->buffer->client->device;
    return NULL;
}

static PJRT_Error *h_Buffer_Memory(PJRT_Buffer_Memory_Args *args) {
    args->memory = &args->buffer->client->memory;
    return NULL;
}

static PJRT_Error *h_Buffer_Delete(PJRT_Buffer_Delete_Args *args) {
    PJRT_Buffer *b = args->buffer;
    if (!b) return NULL;
    b->deleted = true;
    free(b->data);
    b->data = NULL;
    b->nbytes = 0;
    return NULL;
}

static PJRT_Error *h_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args *args) {
    args->is_deleted = args->buffer->deleted;
    return NULL;
}

static PJRT_Error *h_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args *args) {
    PJRT_Buffer *src = args->buffer;
    PJRT_Buffer *dst = buffer_new(src->client, src->type, src->dims,
                                  src->num_dims);
    if (!dst)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "buffer alloc");
    memcpy(dst->data, src->data, src->nbytes);
    args->dst_buffer = dst;
    return NULL;
}

static PJRT_Error *h_Buffer_CopyToMemory(PJRT_Buffer_CopyToMemory_Args *args) {
    PJRT_Buffer *src = args->buffer;
    PJRT_Buffer *dst = buffer_new(src->client, src->type, src->dims,
                                  src->num_dims);
    if (!dst)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "buffer alloc");
    memcpy(dst->data, src->data, src->nbytes);
    args->dst_buffer = dst;
    return NULL;
}

static PJRT_Error *h_Buffer_Bitcast(PJRT_Buffer_Bitcast_Args *args) {
    PJRT_Buffer *src = args->buffer;
    PJRT_Buffer *dst = buffer_new(src->client, args->element_type, args->dims,
                                  args->num_dims);
    if (!dst)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "buffer alloc");
    size_t n = src->nbytes < dst->nbytes ? src->nbytes : dst->nbytes;
    memcpy(dst->data, src->data, n);
    args->out_buffer = dst;
    return NULL;
}

static PJRT_Error *h_Buffer_ToHostBuffer(
    PJRT_Buffer_ToHostBuffer_Args *args) {
    PJRT_Buffer *b = args->src;
    size_t es = type_size(b->type);
    size_t need = b->nbytes;
    if (!args->dst) {
        args->dst_size = need;
        args->event = event_new_ready(PJRT_Error_Code_OK, NULL);
        return NULL;
    }
    if (args->dst_size < need)
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT,
                          "ToHostBuffer: dst %zu B < %zu B", args->dst_size,
                          need);
    if (args->host_layout &&
        args->host_layout->type == PJRT_Buffer_MemoryLayout_Type_Strides &&
        args->host_layout->strides.num_byte_strides == b->num_dims) {
        const int64_t *st = args->host_layout->strides.byte_strides;
        size_t total = 1;
        for (size_t i = 0; i < b->num_dims; i++) total *= (size_t)b->dims[i];
        size_t *idx = calloc(b->num_dims ? b->num_dims : 1, sizeof(size_t));
        if (!idx) {
            args->event = event_new_ready(PJRT_Error_Code_RESOURCE_EXHAUSTED,
                                          "index alloc");
            return NULL;
        }
        for (size_t n = 0; n < total; n++) {
            int64_t off = 0;
            for (size_t d = 0; d < b->num_dims; d++)
                off += (int64_t)idx[d] * st[d];
            memcpy((char *)args->dst + off, (char *)b->data + n * es, es);
            for (size_t d = b->num_dims; d-- > 0;) {
                if (++idx[d] < (size_t)b->dims[d]) break;
                idx[d] = 0;
            }
        }
        free(idx);
    } else {
        memcpy(args->dst, b->data, need);
    }
    args->event = event_new_ready(PJRT_Error_Code_OK, NULL);
    return NULL;
}

static PJRT_Error *h_Buffer_CopyRawToHost(
    PJRT_Buffer_CopyRawToHost_Args *args) {
    PJRT_Buffer *b = args->buffer;
    if (args->offset < 0 || args->transfer_size < 0 ||
        (size_t)(args->offset + args->transfer_size) > b->nbytes)
        return make_error(PJRT_Error_Code_OUT_OF_RANGE,
                          "CopyRawToHost [%lld,+%lld) of %zu B",
                          (long long)args->offset,
                          (long long)args->transfer_size, b->nbytes);
    memcpy(args->dst, (char *)b->data + args->offset,
           (size_t)args->transfer_size);
    args->event = event_new_ready(PJRT_Error_Code_OK, NULL);
    return NULL;
}

static PJRT_Error *h_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args *args) {
    args->is_on_cpu = false;
    return NULL;
}

static PJRT_Error *h_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args *args) {
    args->event = event_new_ready(PJRT_Error_Code_OK, NULL);
    return NULL;
}

static PJRT_Error *h_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args *args) {
    args->buffer_pointer = (uintptr_t)args->buffer->data;
    return NULL;
}

static PJRT_Error *h_Buffer_OpaqueDeviceMemoryDataPointer(
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args *args) {
    args->device_memory_ptr = args->buffer->data;
    return NULL;
}

static PJRT_Error *h_Buffer_IncreaseExternalReferenceCount(
    PJRT_Buffer_IncreaseExternalReferenceCount_Args *args) {
    (void)args;
    return NULL;
}

static PJRT_Error *h_Buffer_DecreaseExternalReferenceCount(
    PJRT_Buffer_DecreaseExternalReferenceCount_Args *args) {
    (void)args;
    return NULL;
}

static PJRT_Error *h_Buffer_DonateWithControlDependency(
    PJRT_Buffer_DonateWithControlDependency_Args *args) {
    /* The engine runs synchronously, so the input is already free to reuse. */
    args->out_buffer = args->buffer;
    args->callback_data = NULL;
    args->dependency_ready_callback = NULL;
    return NULL;
}

/* ----------------------------------------------------------------- compile */

static vc4_buf slot_buf(PJRT_Executable *e, int slot) {
    vc4_buf b;
    size_t bytes = vc4sl_num_elements(&e->prog, slot) * 4;
    if (bytes < 64) bytes = 64;
    b.off = e->slab.off + (uint32_t)e->offs[slot];
    b.bytes = bytes;
    return b;
}

/* Parse the module and give every tensor a fixed slot in one arena slab.
 * Constants and layout tables are written once, here; Execute only uploads
 * the parameters and downloads the results. */
static PJRT_Error *build_program(PJRT_Executable *e, VC4PjrtClient *c) {
    char err[512] = {0};
    if (vc4sl_parse((const uint8_t *)e->code, e->code_size, &e->prog, err,
                    sizeof(err)))
        return make_error(PJRT_Error_Code_UNIMPLEMENTED, "%s", err);
    e->have_prog = 1;
    vc4_ctx *ctx = client_engine(c);
    if (!ctx)
        return make_error(PJRT_Error_Code_UNAVAILABLE,
                          "vc4 engine unavailable: %s",
                          vc4_strerror(c->engine_status));

    size_t nslots = (size_t)e->prog.num_tensors + 1; /* + scratch */
    e->offs = calloc(nslots, sizeof(size_t));
    if (!e->offs)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "slot table");
    size_t total = 0, maxb = 64;
    for (int i = 0; i < e->prog.num_tensors; i++) {
        size_t bytes = vc4sl_num_elements(&e->prog, i) * 4;
        if (bytes < 64) bytes = 64;
        bytes = (bytes + 63) & ~(size_t)63;
        if (bytes > maxb) maxb = bytes;
        e->offs[i] = total;
        total += bytes;
    }
    e->scratch_slot = e->prog.num_tensors;
    e->offs[e->scratch_slot] = total;
    total += (maxb + 63) & ~(size_t)63;
    e->total_bytes = total;
    vc4_status st = vc4_alloc(ctx, total, &e->slab);
    if (st != VC4_OK)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED,
                          "arena: %s (need %zu B)", vc4_strerror(st), total);
    for (int i = 0; i < e->prog.num_tensors; i++) {
        if (!e->prog.tensors[i].is_constant) continue;
        vc4_buf b = slot_buf(e, i);
        st = vc4_upload(ctx, &b, e->prog.tensors[i].const_data,
                        e->prog.tensors[i].const_len);
        if (st != VC4_OK)
            return make_error(PJRT_Error_Code_INTERNAL,
                              "constant upload: %s", vc4_strerror(st));
    }

    e->num_outputs = (size_t)e->prog.num_outputs;
    e->num_parameters = (size_t)e->prog.num_params;
    e->output_types = calloc(e->num_outputs ? e->num_outputs : 1,
                             sizeof(PJRT_Buffer_Type));
    e->output_dims = calloc(e->num_outputs ? e->num_outputs : 1,
                            sizeof(int64_t *));
    e->output_ranks = calloc(e->num_outputs ? e->num_outputs : 1,
                             sizeof(size_t));
    if (!e->output_types || !e->output_dims || !e->output_ranks)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "output metadata");
    for (size_t i = 0; i < e->num_outputs; i++) {
        vc4sl_shape *sh = &e->prog.output_shapes[i];
        e->output_types[i] = PJRT_Buffer_Type_F32;
        e->output_ranks[i] = (size_t)sh->rank;
        if (sh->rank) {
            e->output_dims[i] = malloc(sizeof(int64_t) * (size_t)sh->rank);
            if (!e->output_dims[i])
                return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED,
                                  "output shape");
            memcpy(e->output_dims[i], sh->dims,
                   sizeof(int64_t) * (size_t)sh->rank);
        }
    }
    vlogf("compiled: %d tensors, %d insns, %zu B arena, %zu outputs\n",
          e->prog.num_tensors, e->prog.num_insns, total, e->num_outputs);
    return NULL;
}

static PJRT_Error *h_Client_Compile(PJRT_Client_Compile_Args *args) {
    const PJRT_Program *program = args->program;
    if (!program || !program->code)
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "compile: no code");
    char format[64] = {0};
    if (program->format && program->format_size) {
        size_t n = program->format_size;
        if (n >= sizeof(format)) n = sizeof(format) - 1;
        memcpy(format, program->format, n);
        format[n] = 0;
    }
    vlogf("compile: %zu B, format \"%s\"\n", program->code_size, format);
    const char *dump = getenv("VC4PJRT_DUMP_DIR");
    if (dump && *dump) {
        static int counter;
        char path[1024];
        snprintf(path, sizeof(path), "%s/module_%03d.mlir", dump, counter++);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(program->code, 1, program->code_size, f);
            fclose(f);
            vlogf("compile: module written to %s\n", path);
        }
    }
    PJRT_Executable *e = executable_new();
    if (!e)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "executable alloc");
    e->code = malloc(program->code_size + 1);
    if (!e->code) {
        executable_release(e);
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "module alloc");
    }
    memcpy(e->code, program->code, program->code_size);
    e->code[e->code_size = program->code_size] = 0;
    PJRT_Error *perr = build_program(e, (VC4PjrtClient *)args->client);
    if (perr) {
        executable_release(e);
        return perr;
    }

    PJRT_LoadedExecutable *le = calloc(1, sizeof(*le));
    if (!le) {
        executable_release(e);
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED,
                          "loaded executable alloc");
    }
    le->executable = e;
    le->device = &((VC4PjrtClient *)args->client)->device;
    args->executable = le;
    return NULL;
}

static PJRT_Error *h_Client_Load(PJRT_Client_Load_Args *args) {
    PJRT_Executable *e = (PJRT_Executable *)args->executable;
    if (!e)
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "load: no executable");
    PJRT_LoadedExecutable *le = calloc(1, sizeof(*le));
    if (!le)
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED,
                          "loaded executable alloc");
    e->refcount++;
    le->executable = e;
    le->device = &((VC4PjrtClient *)args->client)->device;
    args->loaded_executable = le;
    return NULL;
}

/* -------------------------------------------------------------- executable */

static PJRT_Error *h_Executable_Destroy(PJRT_Executable_Destroy_Args *args) {
    executable_release((PJRT_Executable *)args->executable);
    return NULL;
}

static PJRT_Error *h_Executable_Name(PJRT_Executable_Name_Args *args) {
    args->executable_name = args->executable->name;
    args->executable_name_size = strlen(args->executable->name);
    return NULL;
}

static PJRT_Error *h_Executable_NumReplicas(
    PJRT_Executable_NumReplicas_Args *args) {
    args->num_replicas = 1;
    return NULL;
}

static PJRT_Error *h_Executable_NumPartitions(
    PJRT_Executable_NumPartitions_Args *args) {
    args->num_partitions = 1;
    return NULL;
}

static PJRT_Error *h_Executable_NumOutputs(
    PJRT_Executable_NumOutputs_Args *args) {
    args->num_outputs = args->executable->num_outputs;
    return NULL;
}

static PJRT_Error *h_Executable_SizeOfGeneratedCode(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args *args) {
    args->size_in_bytes = args->executable->code_size;
    return NULL;
}

static PJRT_Error *h_Executable_Fingerprint(
    PJRT_Executable_Fingerprint_Args *args) {
    args->executable_fingerprint = NULL;
    args->executable_fingerprint_size = 0;
    return NULL;
}

static PJRT_Error *h_Executable_GetCostAnalysis(
    PJRT_Executable_GetCostAnalysis_Args *args) {
    args->num_properties = 0;
    args->properties = NULL;
    return NULL;
}

static PJRT_Error *h_Executable_OutputMemoryKinds(
    PJRT_Executable_OutputMemoryKinds_Args *args) {
    static const char *kinds[64];
    static size_t sizes[64];
    size_t n = args->executable->num_outputs;
    if (n > 64) n = 64;
    for (size_t i = 0; i < n; i++) {
        kinds[i] = "device";
        sizes[i] = strlen("device");
    }
    args->memory_kinds = kinds;
    args->memory_kind_sizes = sizes;
    args->num_outputs = n;
    return NULL;
}

static PJRT_Error *h_Executable_OutputElementTypes(
    PJRT_Executable_OutputElementTypes_Args *args) {
    args->output_types = args->executable->output_types;
    args->num_output_types = args->executable->num_outputs;
    return NULL;
}

static PJRT_Error *h_Executable_OutputDimensions(
    PJRT_Executable_OutputDimensions_Args *args) {
    PJRT_Executable *e = args->executable;
    static int64_t flat[64 * 8];
    static size_t sizes[64];
    size_t off = 0;
    for (size_t i = 0; i < e->num_outputs && i < 64; i++) {
        sizes[i] = e->output_ranks[i];
        for (size_t d = 0; d < e->output_ranks[i] && off < 64 * 8; d++)
            flat[off++] = e->output_dims[i][d];
    }
    args->dims = flat;
    args->dim_sizes = sizes;
    args->num_outputs = e->num_outputs < 64 ? e->num_outputs : 64;
    return NULL;
}

static PJRT_Error *h_Executable_ParameterMemoryKinds(
    PJRT_Executable_ParameterMemoryKinds_Args *args) {
    args->memory_kinds = NULL;
    args->num_parameters = 0;
    return NULL;
}

static PJRT_Error *h_Executable_OptimizedProgram(
    PJRT_Executable_OptimizedProgram_Args *args) {
    PJRT_Executable *e = args->executable;
    PJRT_Program *p = args->program;
    if (!p) return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "no program");
    /* The lowered program is the executable; hand back the module jaxlib
       gave us.  jax uses this for its compilation cache key. */
    p->format = "mlir";
    p->format_size = strlen("mlir");
    if (!p->code) {
        p->code_size = e->code_size;
        return NULL;
    }
    if (p->code_size < e->code_size) {
        p->code_size = e->code_size;
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT,
                          "optimized program buffer is too small");
    }
    memcpy(p->code, e->code, e->code_size);
    p->code_size = e->code_size;
    return NULL;
}

static PJRT_Error *h_Executable_Serialize(PJRT_Executable_Serialize_Args *args) {
    (void)args;
    return make_error(PJRT_Error_Code_UNIMPLEMENTED,
                      "PJRT_Executable_Serialize");
}

static PJRT_Error *h_Executable_DeserializeAndLoad(
    PJRT_Executable_DeserializeAndLoad_Args *args) {
    (void)args;
    return make_error(PJRT_Error_Code_UNIMPLEMENTED,
                      "PJRT_Executable_DeserializeAndLoad");
}

/* ------------------------------------------------------- loaded executable */

static PJRT_Error *h_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args *args) {
    PJRT_LoadedExecutable *le = args->executable;
    if (!le) return NULL;
    executable_release(le->executable);
    free(le);
    return NULL;
}

static PJRT_Error *h_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args *args) {
    args->executable = args->loaded_executable->executable;
    args->executable->refcount++;
    return NULL;
}

static PJRT_Error *h_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args *args) {
    static PJRT_Device *devices[1];
    PJRT_LoadedExecutable *le = args->executable;
    devices[0] = le->device;
    args->addressable_devices = devices;
    args->num_addressable_devices = 1;
    return NULL;
}

static PJRT_Error *h_LoadedExecutable_AddressableDeviceLogicalIds(
    PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args *args) {
    static PJRT_LogicalDeviceIds ids[1] = {{0, 0}};
    args->addressable_device_logical_ids = ids;
    args->num_addressable_device_logical_ids = 1;
    return NULL;
}

static void device_assignment_deleter(PJRT_DeviceAssignmentSerialized *da) {
    free(da);
}

static PJRT_Error *h_LoadedExecutable_GetDeviceAssignment(
    PJRT_LoadedExecutable_GetDeviceAssignment_Args *args) {
    /* One device, one replica, one partition: the empty assignment is the
       serialized form the wrapper treats as "not portable-specific". */
    args->serialized_bytes = NULL;
    args->serialized_bytes_size = 0;
    args->serialized_device_assignment = NULL;
    args->serialized_device_assignment_deleter = device_assignment_deleter;
    return NULL;
}

static PJRT_Error *h_LoadedExecutable_Delete(
    PJRT_LoadedExecutable_Delete_Args *args) {
    args->executable->deleted = true;
    return NULL;
}

static PJRT_Error *h_LoadedExecutable_IsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args *args) {
    args->is_deleted = args->executable->deleted;
    return NULL;
}

static PJRT_Error *run_insn(VC4PjrtClient *c, PJRT_Executable *e,
                            const vc4sl_insn *in) {
    vc4_ctx *ctx = client_engine(c);
    vc4_buf A = slot_buf(e, in->a);
    vc4_buf B = slot_buf(e, in->b);
    vc4_buf D = slot_buf(e, in->dst);
    size_t n = vc4sl_num_elements(&e->prog, in->dst);
    vc4_status st = VC4_OK;
    switch (in->op) {
    case VC4SL_ADD:
        st = vc4_elementwise(ctx, VC4_OP_ADD, &A, &B, &D, n);
        break;
    case VC4SL_MUL:
        st = vc4_elementwise(ctx, VC4_OP_MUL, &A, &B, &D, n);
        break;
    case VC4SL_SUB: {
        vc4_buf S = slot_buf(e, e->scratch_slot);
        st = vc4_unary(ctx, VC4_UOP_NEG, &B, &S, n, 0, 0, 0);
        if (st == VC4_OK)
            st = vc4_elementwise(ctx, VC4_OP_ADD, &A, &S, &D, n);
        break;
    }
    case VC4SL_NEGATE:
        st = vc4_unary(ctx, VC4_UOP_NEG, &A, &D, n, 0, 0, 0);
        break;
    case VC4SL_ABS:
        st = vc4_unary(ctx, VC4_UOP_ABS, &A, &D, n, 0, 0, 0);
        break;
    case VC4SL_AFFINE:
        st = vc4_unary(ctx, VC4_UOP_AFFINE, &A, &D, n, 0.5f, 0.0f, 0);
        break;
    case VC4SL_RELU:
        st = vc4_relu(ctx, &A, &D, n);
        break;
    case VC4SL_COMPARE: {
        static const vc4_op cmp[] = {VC4_OP_CMP_GT, VC4_OP_CMP_GE,
                                     VC4_OP_CMP_LT, VC4_OP_CMP_LE,
                                     VC4_OP_CMP_EQ};
        if (in->aux[0] < 0 || in->aux[0] > 4)
            return make_error(PJRT_Error_Code_UNIMPLEMENTED,
                              "comparison direction %lld",
                              (long long)in->aux[0]);
        st = vc4_elementwise(ctx, cmp[in->aux[0]], &A, &B, &D, n);
        break;
    }
    case VC4SL_DOT:
        st = vc4_matmul(ctx, &A, &B, &D, (uint32_t)in->aux[0],
                        (uint32_t)in->aux[1], (uint32_t)in->aux[2],
                        VC4_LAYOUT_ROW_MAJOR);
        break;
    case VC4SL_DOT_BT:
        st = vc4_matmul_bt(ctx, &A, &B, &D, (uint32_t)in->aux[0],
                           (uint32_t)in->aux[1], (uint32_t)in->aux[2]);
        break;
    case VC4SL_GATHER_HOST: {
        if (in->table < 0 || in->table >= e->prog.num_tables)
            return make_error(PJRT_Error_Code_INTERNAL, "missing layout table");
        size_t src_n = vc4sl_num_elements(&e->prog, in->a);
        float *stage_in = malloc(src_n ? src_n * 4 : 4);
        float *stage_out = malloc(n ? n * 4 : 4);
        if (!stage_in || !stage_out) {
            free(stage_in);
            free(stage_out);
            return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED,
                              "layout staging");
        }
        vc4_status s1 = vc4_download(ctx, &A, stage_in, src_n * 4);
        if (s1 == VC4_OK) {
            const int32_t *tab = e->prog.tables[in->table].idx;
            for (size_t i = 0; i < n; i++) {
                int32_t idx = tab[i];
                if (idx < 0 || (size_t)idx >= src_n) {
                    free(stage_in);
                    free(stage_out);
                    return make_error(PJRT_Error_Code_INTERNAL,
                                      "layout index out of range");
                }
                stage_out[i] = stage_in[idx];
            }
            s1 = vc4_upload(ctx, &D, stage_out, n * 4);
        }
        free(stage_in);
        free(stage_out);
        st = s1;
        break;
    }
    case VC4SL_CONSTANT:
        return NULL;
    default:
        return make_error(PJRT_Error_Code_UNIMPLEMENTED,
                          "%s is outside the vc4 executor", in->name);
    }
    if (st != VC4_OK)
        return make_error(PJRT_Error_Code_INTERNAL, "%s: %s (%s)", in->name,
                          vc4_strerror(st), vc4_last_error(ctx));
    return NULL;
}

static PJRT_Error *h_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args *args) {
    PJRT_LoadedExecutable *le = args->executable;
    PJRT_Executable *e = le->executable;
    if (!e->have_prog)
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT,
                          "executable has no program");
    if (args->num_devices != 1)
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT,
                          "vc4 executes one device, got %zu",
                          args->num_devices);
    if (args->num_args != (size_t)e->prog.num_params)
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT,
                          "expected %d arguments, got %zu",
                          e->prog.num_params, args->num_args);
    VC4PjrtClient *c = le->device->client;
    vc4_ctx *ctx = client_engine(c);
    if (!ctx)
        return make_error(PJRT_Error_Code_UNAVAILABLE, "vc4 engine unavailable");
    PJRT_Buffer *const *in = args->argument_lists[0];
    for (int i = 0; i < e->prog.num_params; i++) {
        vc4_buf b = slot_buf(e, e->prog.param_slots[i]);
        size_t bytes = vc4sl_num_elements(&e->prog, e->prog.param_slots[i]) * 4;
        if (bytes > in[i]->nbytes)
            return make_error(PJRT_Error_Code_INVALID_ARGUMENT,
                              "argument %d is %zu B, needs %zu B", i,
                              in[i]->nbytes, bytes);
        vc4_status st = vc4_upload(ctx, &b, in[i]->data, bytes);
        if (st != VC4_OK)
            return make_error(PJRT_Error_Code_INTERNAL, "input upload: %s",
                              vc4_strerror(st));
    }
    vc4_reset_stats(ctx);
    for (int i = 0; i < e->prog.num_insns; i++) {
        PJRT_Error *err = run_insn(c, e, &e->prog.insns[i]);
        if (err) return err;
    }
    vc4_stats stats;
    vc4_get_stats(ctx, &stats);
    vlogf("execute: %d insns, %" PRIu64 " launches, %" PRIu64 " ns kernels\n",
          e->prog.num_insns, stats.launches, stats.kernel_ns);
    for (size_t i = 0; i < e->num_outputs; i++) {
        PJRT_Buffer *out = buffer_new(c, PJRT_Buffer_Type_F32,
                                      e->output_dims[i], e->output_ranks[i]);
        if (!out)
            return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED,
                              "output buffer alloc");
        vc4_buf b = slot_buf(e, e->prog.output_slots[i]);
        size_t bytes = vc4sl_num_elements(&e->prog, e->prog.output_slots[i]) * 4;
        vc4_status st = vc4_download(ctx, &b, out->data, bytes);
        if (st != VC4_OK) {
            buffer_free(out);
            return make_error(PJRT_Error_Code_INTERNAL, "output download: %s",
                              vc4_strerror(st));
        }
        args->output_lists[0][i] = out;
    }
    if (args->device_complete_events)
        args->device_complete_events[0] = event_new_ready(PJRT_Error_Code_OK,
                                                          NULL);
    return NULL;
}

/* --------------------------------------------------------------- API table */

static PJRT_Api g_api = {
    sizeof(PJRT_Api),
    NULL,
    {
        PJRT_Api_Version_STRUCT_SIZE,
        NULL,
        PJRT_API_MAJOR,
        PJRT_API_MINOR,
    },
    .PJRT_Error_Destroy = h_Error_Destroy,
    .PJRT_Error_Message = h_Error_Message,
    .PJRT_Error_GetCode = h_Error_GetCode,
    .PJRT_Error_ForEachPayload = h_Error_ForEachPayload,

    .PJRT_Plugin_Initialize = h_Plugin_Initialize,
    .PJRT_Plugin_Attributes = h_Plugin_Attributes,

    .PJRT_Event_Destroy = h_Event_Destroy,
    .PJRT_Event_IsReady = h_Event_IsReady,
    .PJRT_Event_Error = h_Event_Error,
    .PJRT_Event_Await = h_Event_Await,
    .PJRT_Event_OnReady = h_Event_OnReady,
    .PJRT_Event_Create = h_Event_Create,
    .PJRT_Event_Set = h_Event_Set,

    .PJRT_Client_Create = h_Client_Create,
    .PJRT_Client_Destroy = h_Client_Destroy,
    .PJRT_Client_PlatformName = h_Client_PlatformName,
    .PJRT_Client_ProcessIndex = h_Client_ProcessIndex,
    .PJRT_Client_PlatformVersion = h_Client_PlatformVersion,
    .PJRT_Client_Devices = h_Client_Devices,
    .PJRT_Client_AddressableDevices = h_Client_AddressableDevices,
    .PJRT_Client_LookupDevice = h_Client_LookupDevice,
    .PJRT_Client_LookupAddressableDevice = h_Client_LookupAddressableDevice,
    .PJRT_Client_AddressableMemories = h_Client_AddressableMemories,
    .PJRT_Client_Compile = h_Client_Compile,
    .PJRT_Client_DefaultDeviceAssignment = h_Client_DefaultDeviceAssignment,
    .PJRT_Client_BufferFromHostBuffer = h_Client_BufferFromHostBuffer,
    .PJRT_Client_CreateUninitializedBuffer = h_Client_CreateUninitializedBuffer,
    .PJRT_Client_TopologyDescription = h_Client_TopologyDescription,

    .PJRT_DeviceDescription_Id = h_Description_Id,
    .PJRT_DeviceDescription_ProcessIndex = h_Description_ProcessIndex,
    .PJRT_DeviceDescription_Attributes = h_Description_Attributes,
    .PJRT_DeviceDescription_Kind = h_Description_Kind,
    .PJRT_DeviceDescription_DebugString = h_Description_DebugString,
    .PJRT_DeviceDescription_ToString = h_Description_ToString,

    .PJRT_Device_GetDescription = h_Device_GetDescription,
    .PJRT_Device_IsAddressable = h_Device_IsAddressable,
    .PJRT_Device_LocalHardwareId = h_Device_LocalHardwareId,
    .PJRT_Device_AddressableMemories = h_Device_AddressableMemories,
    .PJRT_Device_DefaultMemory = h_Device_DefaultMemory,
    .PJRT_Device_MemoryStats = h_Device_MemoryStats,

    .PJRT_Memory_Id = h_Memory_Id,
    .PJRT_Memory_Kind = h_Memory_Kind,
    .PJRT_Memory_Kind_Id = h_Memory_Kind_Id,
    .PJRT_Memory_DebugString = h_Memory_DebugString,
    .PJRT_Memory_ToString = h_Memory_ToString,
    .PJRT_Memory_AddressableByDevices = h_Memory_AddressableByDevices,

    .PJRT_Executable_Destroy = h_Executable_Destroy,
    .PJRT_Executable_Name = h_Executable_Name,
    .PJRT_Executable_NumReplicas = h_Executable_NumReplicas,
    .PJRT_Executable_NumPartitions = h_Executable_NumPartitions,
    .PJRT_Executable_NumOutputs = h_Executable_NumOutputs,
    .PJRT_Executable_SizeOfGeneratedCodeInBytes =
        h_Executable_SizeOfGeneratedCode,
    .PJRT_Executable_Fingerprint = h_Executable_Fingerprint,
    .PJRT_Executable_GetCostAnalysis = h_Executable_GetCostAnalysis,
    .PJRT_Executable_OutputMemoryKinds = h_Executable_OutputMemoryKinds,
    .PJRT_Executable_OutputElementTypes = h_Executable_OutputElementTypes,
    .PJRT_Executable_OutputDimensions = h_Executable_OutputDimensions,
    .PJRT_Executable_ParameterMemoryKinds = h_Executable_ParameterMemoryKinds,
    .PJRT_Executable_OptimizedProgram = h_Executable_OptimizedProgram,
    .PJRT_Executable_Serialize = h_Executable_Serialize,
    .PJRT_Executable_DeserializeAndLoad = h_Executable_DeserializeAndLoad,

    .PJRT_LoadedExecutable_Destroy = h_LoadedExecutable_Destroy,
    .PJRT_LoadedExecutable_GetExecutable = h_LoadedExecutable_GetExecutable,
    .PJRT_LoadedExecutable_AddressableDevices =
        h_LoadedExecutable_AddressableDevices,
    .PJRT_LoadedExecutable_AddressableDeviceLogicalIds =
        h_LoadedExecutable_AddressableDeviceLogicalIds,
    .PJRT_LoadedExecutable_Delete = h_LoadedExecutable_Delete,
    .PJRT_LoadedExecutable_IsDeleted = h_LoadedExecutable_IsDeleted,
    .PJRT_LoadedExecutable_GetDeviceAssignment =
        h_LoadedExecutable_GetDeviceAssignment,
    .PJRT_LoadedExecutable_Execute = h_LoadedExecutable_Execute,

    .PJRT_Client_Load = h_Client_Load,

    .PJRT_Buffer_Destroy = h_Buffer_Destroy,
    .PJRT_Buffer_ElementType = h_Buffer_ElementType,
    .PJRT_Buffer_Dimensions = h_Buffer_Dimensions,
    .PJRT_Buffer_UnpaddedDimensions = h_Buffer_UnpaddedDimensions,
    .PJRT_Buffer_DynamicDimensionIndices = h_Buffer_DynamicDimensionIndices,
    .PJRT_Buffer_GetMemoryLayout = h_Buffer_GetMemoryLayout,
    .PJRT_Buffer_OnDeviceSizeInBytes = h_Buffer_OnDeviceSizeInBytes,
    .PJRT_Buffer_Device = h_Buffer_Device,
    .PJRT_Buffer_Memory = h_Buffer_Memory,
    .PJRT_Buffer_Delete = h_Buffer_Delete,
    .PJRT_Buffer_IsDeleted = h_Buffer_IsDeleted,
    .PJRT_Buffer_CopyToDevice = h_Buffer_CopyToDevice,
    .PJRT_Buffer_CopyToMemory = h_Buffer_CopyToMemory,
    .PJRT_Buffer_Bitcast = h_Buffer_Bitcast,
    .PJRT_Buffer_ToHostBuffer = h_Buffer_ToHostBuffer,
    .PJRT_Buffer_CopyRawToHost = h_Buffer_CopyRawToHost,
    .PJRT_Buffer_IsOnCpu = h_Buffer_IsOnCpu,
    .PJRT_Buffer_ReadyEvent = h_Buffer_ReadyEvent,
    .PJRT_Buffer_UnsafePointer = h_Buffer_UnsafePointer,
    .PJRT_Buffer_OpaqueDeviceMemoryDataPointer =
        h_Buffer_OpaqueDeviceMemoryDataPointer,
    .PJRT_Buffer_IncreaseExternalReferenceCount =
        h_Buffer_IncreaseExternalReferenceCount,
    .PJRT_Buffer_DecreaseExternalReferenceCount =
        h_Buffer_DecreaseExternalReferenceCount,
    .PJRT_Buffer_DonateWithControlDependency =
        h_Buffer_DonateWithControlDependency,

    .PJRT_TopologyDescription_Create = h_Topology_Create,
    .PJRT_TopologyDescription_Destroy = h_Topology_Destroy,
    .PJRT_TopologyDescription_PlatformName = h_Topology_PlatformName,
    .PJRT_TopologyDescription_PlatformVersion = h_Topology_PlatformVersion,
    .PJRT_TopologyDescription_GetDeviceDescriptions =
        h_Topology_GetDeviceDescriptions,
    .PJRT_TopologyDescription_Serialize = h_Topology_Serialize,
    .PJRT_TopologyDescription_Deserialize = h_Topology_Deserialize,
    .PJRT_TopologyDescription_Attributes = h_Topology_Attributes,
    .PJRT_TopologyDescription_Fingerprint = h_Topology_Fingerprint,
};

PJRT_Api *GetPjrtApi(void) {
    vlogf("GetPjrtApi: returning PJRT API %d.%d\n", PJRT_API_MAJOR,
          PJRT_API_MINOR);
    return &g_api;
}
