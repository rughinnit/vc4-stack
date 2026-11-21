// vc4_ep.cc -- a capability-only ONNX Runtime execution-provider plugin for
// the VideoCore IV QPUs.
//
// This library exists to answer one question with code instead of argument:
// can an execution provider for this board be built and registered against the
// ONNX Runtime that is installed (1.32.0, cp313, manylinux aarch64)? It can.
// The plugin is compiled against the vendored ORT 1.32.0 plugin headers in
// ../../third_party/onnxruntime-1.32.0, registers a factory, advertises the
// VideoCore IV as an EP device, and claims no nodes.
//
// It does not execute anything. GetCapability returns success without adding a
// single node to the graph support info, so ORT's partitioner leaves every node
// on the EP it would otherwise use and the session output is identical to the
// CPU EP. Claiming a node here would be a lie: no compute path through
// libvc4infer is wired into this plugin yet.
//
// What it establishes:
//   * the plugin ABI in the installed wheel is usable with the published
//     header for the same version (the wheel itself ships no headers);
//   * CreateEpFactories / ReleaseEpFactory are resolved and called;
//   * OrtEpApi::CreateHardwareDevice and CreateEpDevice accept a device that
//     ORT did not discover by itself;
//   * a session can be created with that device and produces CPU results.
//
// One ABI detail is not optional despite the header calling it optional:
// ORT 1.32 calls OrtEpFactory::CreateDataTransfer during library registration
// and does not check for NULL. Omitting it segfaults the process at
// register_execution_provider_library(); the implementation below returns an
// empty result ("no device memory"), which is the truth for this EP.
//
// Build: make -C infer/onnxruntime_ep/c_ep
// Register: onnxruntime.register_execution_provider_library("vc4", path)

#include <onnxruntime_c_api.h>
#include <onnxruntime_ep_device_ep_metadata_keys.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

namespace {

constexpr uint32_t kBroadcomVendorId = 0x14E4;
constexpr uint32_t kVc4DeviceId = 0x0000;

const char* const kEpName = "vc4";
const char* const kEpVendor = "vc4infer";
const char* const kEpVersion = "0.1.0-capability-only";

// Set by CreateEpFactories. One process registers one copy of this library in
// the intended use; a second registration overwrites the pointers with the
// same api objects.
const OrtApi* g_api = nullptr;
const OrtEpApi* g_ep_api = nullptr;
const OrtLogger* g_logger = nullptr;

unsigned long g_capability_calls = 0;
unsigned long g_nodes_examined = 0;
unsigned long g_nodes_claimed = 0;

// Created inside GetSupportedDevices and owned until ReleaseEpFactory: the
// OrtEpDevice keeps a reference to the hardware device, so it has to outlive
// the factory call that created it.
OrtHardwareDevice* g_hw_device = nullptr;

struct Vc4Factory {
  OrtEpFactory base;  // must stay first: ORT hands back a base pointer
};

struct Vc4Ep {
  OrtEp base;         // must stay first
};

void log_line(const char* text) {
  std::fprintf(stderr, "vc4_ep: %s\n", text);
  if (g_logger != nullptr && g_api != nullptr) {
    OrtStatus* status = g_api->Logger_LogMessage(
        g_logger, ORT_LOGGING_LEVEL_INFO, text, ORT_FILE, __LINE__,
        __FUNCTION__);
    if (status != nullptr) {
      g_api->ReleaseStatus(status);
    }
  }
}

OrtStatus* not_implemented(const char* what) {
  return g_api->CreateStatus(ORT_NOT_IMPLEMENTED, what);
}

// ------------------------------------------------------------------- factory
const char* ORT_API_CALL FactoryGetName(const OrtEpFactory*) NO_EXCEPTION {
  return kEpName;
}

const char* ORT_API_CALL FactoryGetVendor(const OrtEpFactory*) NO_EXCEPTION {
  return kEpVendor;
}

uint32_t ORT_API_CALL FactoryGetVendorId(const OrtEpFactory*) NO_EXCEPTION {
  return kBroadcomVendorId;
}

const char* ORT_API_CALL FactoryGetVersion(const OrtEpFactory*) NO_EXCEPTION {
  return kEpVersion;
}

OrtStatus* ORT_API_CALL FactoryGetSupportedDevices(
    OrtEpFactory* factory, const OrtHardwareDevice* const* devices,
    size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
    size_t* num_ep_devices) NO_EXCEPTION {
  (void)devices;      // ORT's own discovery finds a CPU and nothing else here
  (void)num_devices;
  *num_ep_devices = 0;
  if (g_ep_api == nullptr) {
    return g_api->CreateStatus(ORT_FAIL, "vc4_ep: OrtEpApi is unavailable");
  }
  if (max_ep_devices == 0) {
    return g_api->CreateStatus(ORT_FAIL, "vc4_ep: no room for an EP device");
  }
  if (g_hw_device == nullptr) {
    OrtStatus* status = g_ep_api->CreateHardwareDevice(
        OrtHardwareDeviceType_GPU, kBroadcomVendorId, kVc4DeviceId,
        "Broadcom VideoCore IV", nullptr, &g_hw_device);
    if (status != nullptr) {
      return status;
    }
  }

  OrtKeyValuePairs* metadata = nullptr;
  g_api->CreateKeyValuePairs(&metadata);
  g_api->AddKeyValuePair(metadata, "vc4.mode", "capability-only");
  g_api->AddKeyValuePair(metadata, "vc4.execution", "none: claims 0 nodes");
  g_api->AddKeyValuePair(metadata, "vc4.driver", "vc4d");

  OrtStatus* status = g_ep_api->CreateEpDevice(
      factory, g_hw_device, metadata, nullptr, &ep_devices[0]);
  g_api->ReleaseKeyValuePairs(metadata);
  if (status != nullptr) {
    return status;
  }
  *num_ep_devices = 1;
  log_line("EP device registered: Broadcom VideoCore IV (GPU), "
           "mode=capability-only");
  return nullptr;
}

// No device memory: tensors cross this EP boundary only if it ever claims a
// node, and it does not. Returning an empty result here is required, not
// optional -- see the note at the top of the file.
OrtStatus* ORT_API_CALL FactoryCreateDataTransfer(
    OrtEpFactory*, OrtDataTransferImpl** data_transfer) NO_EXCEPTION {
  *data_transfer = nullptr;
  return nullptr;
}

bool ORT_API_CALL FactoryIsStreamAware(const OrtEpFactory*) NO_EXCEPTION {
  return false;  // ORT calls this when a session is built
}

OrtStatus* ORT_API_CALL FactoryGetNumCustomOpDomains(
    OrtEpFactory*, size_t* num_domains) NO_EXCEPTION {
  *num_domains = 0;  // no kernels, therefore no custom operator domains
  return nullptr;
}

OrtStatus* ORT_API_CALL FactoryGetCustomOpDomains(
    OrtEpFactory*, OrtCustomOpDomain** domains,
    size_t num_domains) NO_EXCEPTION {
  (void)domains;
  (void)num_domains;
  return nullptr;
}

OrtStatus* ORT_API_CALL FactoryValidateCompiledModelCompatibilityInfo(
    OrtEpFactory*, const OrtHardwareDevice* const*, size_t, const char*,
    OrtCompiledModelCompatibility*) NO_EXCEPTION {
  return not_implemented("vc4_ep: no compiled models");
}

OrtStatus* ORT_API_CALL FactoryCreateAllocator(
    OrtEpFactory*, const OrtMemoryInfo*, const OrtKeyValuePairs*,
    OrtAllocator** allocator) NO_EXCEPTION {
  *allocator = nullptr;  // ORT's default CPU allocator stays in charge
  return nullptr;
}

void ORT_API_CALL FactoryReleaseAllocator(
    OrtEpFactory*, OrtAllocator*) NO_EXCEPTION {}

OrtStatus* ORT_API_CALL FactoryCreateSyncStreamForDevice(
    OrtEpFactory*, const OrtMemoryDevice*, const OrtKeyValuePairs*,
    OrtSyncStreamImpl** stream) NO_EXCEPTION {
  *stream = nullptr;
  return nullptr;
}

OrtStatus* ORT_API_CALL FactoryGetHardwareDeviceIncompatibilityDetails(
    OrtEpFactory*, const OrtHardwareDevice*,
    OrtDeviceEpIncompatibilityDetails*) NO_EXCEPTION {
  return not_implemented("vc4_ep: no incompatibility model");
}

OrtStatus* ORT_API_CALL FactoryCreateExternalResourceImporterForDevice(
    OrtEpFactory*, const OrtEpDevice*,
    OrtExternalResourceImporterImpl** out_importer) NO_EXCEPTION {
  *out_importer = nullptr;
  return nullptr;
}

OrtStatus* ORT_API_CALL FactoryInitGraphicsInterop(
    OrtEpFactory*, const OrtEpDevice*,
    const OrtGraphicsInteropConfig*) NO_EXCEPTION {
  return not_implemented("vc4_ep: no graphics interop");
}

OrtStatus* ORT_API_CALL FactoryDeinitGraphicsInterop(
    OrtEpFactory*, const OrtEpDevice*) NO_EXCEPTION {
  return nullptr;
}

OrtStatus* ORT_API_CALL FactorySelectBestModelCandidate(
    OrtEpFactory*, const OrtHardwareDevice*, const OrtKeyValuePairs* const*,
    size_t, const OrtSessionOptions*, size_t*) NO_EXCEPTION {
  return not_implemented("vc4_ep: no compiled model candidates");
}

OrtStatus* ORT_API_CALL FactoryGetDeviceMemoryCapacity(
    OrtEpFactory*, const OrtEpDevice*, OrtDeviceMemoryType,
    uint64_t*, uint64_t*) NO_EXCEPTION {
  return not_implemented("vc4_ep: no device memory to report");
}

// --------------------------------------------------------------------- OrtEp
const char* ORT_API_CALL EpGetName(const OrtEp*) NO_EXCEPTION {
  return kEpName;
}

OrtStatus* ORT_API_CALL EpGetCapability(
    OrtEp*, const OrtGraph* graph,
    OrtEpGraphSupportInfo* support_info) NO_EXCEPTION {
  (void)support_info;
  ++g_capability_calls;
  size_t num_nodes = 0;
  OrtStatus* status = g_api->Graph_GetNumNodes(graph, &num_nodes);
  if (status != nullptr) {
    return status;
  }
  g_nodes_examined += num_nodes;
  // Deliberately empty: no node has an execute path in this build, so ORT
  // keeps its own placement. The count is logged, not claimed.
  std::fprintf(stderr,
               "vc4_ep: GetCapability on %zu node(s), claimed 0 "
               "(capability-only build)\n",
               num_nodes);
  return nullptr;
}

OrtStatus* ORT_API_CALL FactoryCreateEp(
    OrtEpFactory*, const OrtHardwareDevice* const* devices,
    const OrtKeyValuePairs* const* ep_metadata_pairs, size_t num_devices,
    const OrtSessionOptions*, const OrtLogger*,
    OrtEp** out) NO_EXCEPTION {
  (void)devices;
  (void)ep_metadata_pairs;
  (void)num_devices;
  if (out == nullptr) {
    return g_api->CreateStatus(ORT_INVALID_ARGUMENT,
                               "vc4_ep: OrtEp output pointer is null");
  }
  auto* ep = new (std::nothrow) Vc4Ep();
  if (ep == nullptr) {
    return g_api->CreateStatus(ORT_FAIL, "vc4_ep: out of memory");
  }
  std::memset(&ep->base, 0, sizeof(ep->base));
  ep->base.ort_version_supported = ORT_API_VERSION;
  ep->base.GetName = EpGetName;
  ep->base.GetCapability = EpGetCapability;
  *out = &ep->base;
  log_line("session EP created; every node stays where ORT put it");
  return nullptr;
}

void ORT_API_CALL FactoryReleaseEp(OrtEpFactory*, OrtEp* ep) NO_EXCEPTION {
  delete reinterpret_cast<Vc4Ep*>(ep);
}

}  // namespace

// Exported counters for tests and tools. These describe this build, not a
// promise: g_nodes_claimed stays zero until an execute path exists.
extern "C" {

unsigned long vc4_ep_capability_calls(void) { return g_capability_calls; }
unsigned long vc4_ep_nodes_examined(void) { return g_nodes_examined; }
unsigned long vc4_ep_nodes_claimed(void) { return g_nodes_claimed; }

const char* vc4_ep_status(void) {
  return "capability-only: registers an EP device, claims 0 nodes, executes "
         "nothing";
}

OrtStatus* ORT_API_CALL CreateEpFactories(
    const char* registered_name, const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger, OrtEpFactory** factories,
    size_t max_factories, size_t* num_factories) {
  if (num_factories == nullptr || factories == nullptr) {
    return nullptr;  // no OrtApi yet, so no status can be reported
  }
  *num_factories = 0;
  const OrtApi* api = ort_api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) {
    // The runtime does not offer the API version this library was compiled
    // against. Registering nothing is the honest outcome; there is no OrtApi
    // with which to build an error status the caller could read.
    return nullptr;
  }
  g_api = api;
  g_ep_api = api->GetEpApi();
  g_logger = default_logger;
  if (g_ep_api == nullptr) {
    return api->CreateStatus(ORT_FAIL, "vc4_ep: OrtEpApi is unavailable");
  }
  if (max_factories == 0) {
    return api->CreateStatus(ORT_FAIL, "vc4_ep: no room for an EP factory");
  }
  auto* factory = new (std::nothrow) Vc4Factory();
  if (factory == nullptr) {
    return api->CreateStatus(ORT_FAIL, "vc4_ep: out of memory");
  }
  std::memset(&factory->base, 0, sizeof(factory->base));
  factory->base.ort_version_supported = ORT_API_VERSION;
  factory->base.GetName = FactoryGetName;
  factory->base.GetVendor = FactoryGetVendor;
  factory->base.GetVendorId = FactoryGetVendorId;
  factory->base.GetVersion = FactoryGetVersion;
  factory->base.GetSupportedDevices = FactoryGetSupportedDevices;
  factory->base.CreateEp = FactoryCreateEp;
  factory->base.ReleaseEp = FactoryReleaseEp;
  factory->base.CreateDataTransfer = FactoryCreateDataTransfer;
  factory->base.IsStreamAware = FactoryIsStreamAware;
  factory->base.GetNumCustomOpDomains = FactoryGetNumCustomOpDomains;
  factory->base.GetCustomOpDomains = FactoryGetCustomOpDomains;
  factory->base.ValidateCompiledModelCompatibilityInfo =
      FactoryValidateCompiledModelCompatibilityInfo;
  factory->base.CreateAllocator = FactoryCreateAllocator;
  factory->base.ReleaseAllocator = FactoryReleaseAllocator;
  factory->base.CreateSyncStreamForDevice = FactoryCreateSyncStreamForDevice;
  factory->base.GetHardwareDeviceIncompatibilityDetails =
      FactoryGetHardwareDeviceIncompatibilityDetails;
  factory->base.CreateExternalResourceImporterForDevice =
      FactoryCreateExternalResourceImporterForDevice;
  factory->base.InitGraphicsInterop = FactoryInitGraphicsInterop;
  factory->base.DeinitGraphicsInterop = FactoryDeinitGraphicsInterop;
  factory->base.SelectBestModelCandidate = FactorySelectBestModelCandidate;
  factory->base.GetDeviceMemoryCapacity = FactoryGetDeviceMemoryCapacity;
  factories[0] = &factory->base;
  *num_factories = 1;
  std::fprintf(stderr,
               "vc4_ep: CreateEpFactories(\"%s\") against ORT API %d; %s\n",
               registered_name != nullptr ? registered_name : "",
               ORT_API_VERSION, vc4_ep_status());
  return nullptr;
}

OrtStatus* ORT_API_CALL ReleaseEpFactory(OrtEpFactory* factory) {
  std::fprintf(stderr,
               "vc4_ep: ReleaseEpFactory: %lu capability query(ies), "
               "%lu node(s) examined, %lu claimed\n",
               g_capability_calls, g_nodes_examined, g_nodes_claimed);
  if (g_hw_device != nullptr && g_ep_api != nullptr) {
    g_ep_api->ReleaseHardwareDevice(g_hw_device);
    g_hw_device = nullptr;
  }
  delete reinterpret_cast<Vc4Factory*>(factory);
  return nullptr;
}

}  // extern "C"
