#ifndef OFFLINE_STREAM_H
#define OFFLINE_STREAM_H

#include <map>
#include <memory>
#include <string>

#include "campplus-model.h"
#include "model.h"
#include "punc-model.h"
#include "vad-model.h"
#if !defined(__APPLE__)
#include "com-define.h"
#include "itn-model.h"
#endif

namespace funasr {
class OfflineStream {
 public:
  OfflineStream(std::map<std::string, std::string>& model_path, int thread_num,
                bool use_gpu = false, int batch_size = 1);
  ~OfflineStream() {};

  std::unique_ptr<VadModel> vad_handle = nullptr;
  std::unique_ptr<Model> asr_handle = nullptr;
  std::unique_ptr<PuncModel> punc_handle = nullptr;
#if !defined(__APPLE__)
  std::unique_ptr<ITNModel> itn_handle = nullptr;
#endif
  bool UseVad() { return use_vad; };
  bool UsePunc() { return use_punc; };
  bool UseITN() { return use_itn; };
  std::string GetModelType() { return model_type; };

 private:
  bool use_vad = false;
  bool use_punc = false;
  bool use_itn = false;
  std::string model_type = MODEL_PARA;
};

OfflineStream* CreateOfflineStream(
    std::map<std::string, std::string>& model_path, int thread_num = 1,
    bool use_gpu = false, int batch_size = 1);
}  // namespace funasr
#endif
