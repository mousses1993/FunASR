#pragma once
// system
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <win_func.h>
#else
#include <unistd.h>
#endif

using namespace std;
// third part
#if defined(__APPLE__)
#include <onnxruntime/onnxruntime_cxx_api.h>
#else
#include "itn-model.h"
#include "itn-processor.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_run_options_config_keys.h"
#endif

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "kaldi/decoder/lattice-faster-online-decoder.h"
// mine
#include <glog/logging.h>

#include "audio.h"
#include "com-define.h"
#include "common-struct.h"
#include "commonfunc.h"
#include "ct-transformer-online.h"
#include "ct-transformer.h"
#include "e2e-vad.h"
#include "encode_converter.h"
#include "fsmn-vad-online.h"
#include "fsmn-vad.h"
#include "model.h"
#include "paraformer.h"
#include "phone-set.h"
#include "predefine-coe.h"
#include "punc-model.h"
#include "resample.h"
#include "seg_dict.h"
#include "sensevoice-small.h"
#include "tensor.h"
#include "tokenizer.h"
#include "util.h"
#include "vad-model.h"
#include "vocab.h"
#include "wfst-decoder.h"
#ifdef USE_GPU
#include "paraformer-torch.h"
#endif
#include "campplus-model.h"
#include "funasrruntime.h"
#include "offline-stream.h"
#include "paraformer-online.h"
#include "speaker-diarization.h"
#include "tpass-online-stream.h"
#include "tpass-stream.h"
