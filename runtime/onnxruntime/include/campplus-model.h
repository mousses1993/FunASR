/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 *
 * CAMPPlus: CAM++ Speaker Embedding Model
 * Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)
 */

#ifndef CAMPPLUS_MODEL_H
#define CAMPPLUS_MODEL_H

#include <onnxruntime_cxx_api.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace funasr {

// Speaker embedding dimension for CAM++ model
constexpr int SPEAKER_EMBEDDING_DIM = 192;
constexpr int SPEAKER_FBANK_DIM = 80;

/**
 * SpeakerSegment represents a speaker segment with time information
 * [start_time, end_time, speaker_id]
 */
struct SpeakerSegment {
  float start_time;  // Start time in seconds
  float end_time;    // End time in seconds
  int speaker_id;    // Speaker label (0, 1, 2, ...)
};

/**
 * CAMPPlusModel: CAM++ Speaker Embedding Model
 * Used for extracting speaker embeddings from audio segments
 */
class CAMPPlusModel {
 public:
  CAMPPlusModel();
  ~CAMPPlusModel();

  /**
   * Initialize the CAM++ model
   * @param model_path Path to the ONNX model file
   * @param config_path Path to the config file (optional)
   * @param thread_num Number of threads for inference
   * @return true if successful, false otherwise
   */
  bool InitModel(const std::string &model_path,
                 const std::string &config_path = "", int thread_num = 1);

  /**
   * Extract speaker embedding from audio features
   * @param features Input fbank features [num_frames, feat_dim]
   * @param num_frames Number of frames
   * @param feat_dim Feature dimension (should be 80)
   * @return Speaker embedding vector [embedding_dim] (192 for CAM++)
   */
  std::vector<float> ExtractEmbedding(const float *features, int num_frames,
                                      int feat_dim = SPEAKER_FBANK_DIM);

  /**
   * Extract speaker embeddings from multiple audio segments
   * @param audio_segments List of audio segments, each is [start, end,
   * audio_data]
   * @return List of speaker embeddings
   */
  std::vector<std::vector<float>> ExtractEmbeddings(
      const std::vector<std::tuple<float, float, std::vector<float>>>
          &audio_segments);

  /**
   * Get the sample rate expected by the model
   */
  int GetSampleRate() const { return sample_rate_; }

  /**
   * Get the embedding dimension
   */
  int GetEmbeddingDim() const { return SPEAKER_EMBEDDING_DIM; }

  /**
   * Check if model is initialized
   */
  bool IsInitialized() const { return initialized_; }

 private:
  // ONNX Runtime session
  std::shared_ptr<Ort::Session> session_ = nullptr;
  Ort::Env env_;
  Ort::SessionOptions session_options_;

  // Input/Output names
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char *> input_name_ptrs_;
  std::vector<const char *> output_name_ptrs_;

  // Model configuration
  int sample_rate_ = 16000;
  int thread_num_ = 1;
  bool initialized_ = false;

  // Helper functions
  void LoadConfigFromYaml(const std::string &config_path);
  std::vector<float> PreprocessAudio(const std::vector<float> &audio);
};

/**
 * Factory function to create CAMPPlus model
 */
CAMPPlusModel *CreateCAMPPlusModel(
    const std::map<std::string, std::string> &model_path, int thread_num = 1);

}  // namespace funasr

#endif  // CAMPPLUS_MODEL_H
