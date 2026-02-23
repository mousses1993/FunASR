/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 *
 * Speaker Diarization Module
 * Performs speaker clustering and segmentation
 */

#ifndef SPEAKER_DIARIZATION_H
#define SPEAKER_DIARIZATION_H

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "campplus-model.h"
#include "funasrruntime.h"

namespace funasr {

// Default parameters for speaker diarization
constexpr float DEFAULT_SEGMENT_DURATION =
    1.5f;  // Duration of each segment in seconds
constexpr float DEFAULT_SEGMENT_SHIFT =
    0.75f;  // Shift between segments in seconds
constexpr int DEFAULT_MIN_NUM_SPEAKERS = 1;
constexpr int DEFAULT_MAX_NUM_SPEAKERS = 15;
constexpr float DEFAULT_PVAL = 0.022f;  // P-value for spectral clustering
constexpr float DEFAULT_MERGE_THRESHOLD =
    0.78f;  // Cosine similarity threshold for merging
constexpr float DEFAULT_MIN_SEGMENT_DURATION =
    0.7f;  // Minimum segment duration in seconds

/**
 * SpectralClustering: Spectral clustering for speaker diarization
 * Adapted from speechbrain implementation
 */
class SpectralClustering {
 public:
  SpectralClustering(int min_num_spks = DEFAULT_MIN_NUM_SPEAKERS,
                     int max_num_spks = DEFAULT_MAX_NUM_SPEAKERS,
                     float pval = DEFAULT_PVAL);

  /**
   * Perform spectral clustering on embeddings
   * @param embeddings Input embeddings [N, D] where N is number of segments
   * @param oracle_num Optional known number of speakers (for evaluation)
   * @return Cluster labels for each segment
   */
  std::vector<int> Cluster(const std::vector<std::vector<float>>& embeddings,
                           int oracle_num = -1);

 private:
  int min_num_spks_;
  int max_num_spks_;
  float pval_;

  // Compute similarity matrix
  std::vector<std::vector<float>> ComputeSimilarityMatrix(
      const std::vector<std::vector<float>>& embeddings);

  // P-pruning of similarity matrix
  void PPruning(std::vector<std::vector<float>>& sim_matrix);

  // Compute Laplacian matrix
  std::vector<std::vector<float>> ComputeLaplacian(
      std::vector<std::vector<float>>& sym_matrix);

  // Get spectral embeddings
  std::pair<std::vector<std::vector<float>>, int> GetSpectralEmbeddings(
      const std::vector<std::vector<float>>& laplacian, int oracle_num);

  // K-means clustering
  std::vector<int> KMeansClustering(
      const std::vector<std::vector<float>>& embeddings, int k);
};

/**
 * SpeakerDiarization: Main class for speaker diarization
 */
class SpeakerDiarization {
 public:
  SpeakerDiarization();
  ~SpeakerDiarization();

  /**
   * Initialize the speaker diarization system
   * @param campplus_model Initialized CAM++ model for speaker embedding
   *                       Note: The caller owns this pointer and must ensure it
   * remains valid for the lifetime of this SpeakerDiarization instance.
   * @param config Configuration parameters
   */
  bool Init(CAMPPlusModel* campplus_model,
            const std::map<std::string, std::string>& config = {});

  /**
   * Perform speaker diarization on audio segments
   * @param vad_segments VAD segments [start_time, end_time, audio_data]
   * @param sample_rate Audio sample rate
   * @return Speaker segments with speaker labels
   */
  std::vector<SpeakerSegment> Process(
      const std::vector<std::tuple<float, float, std::vector<float>>>&
          vad_segments,
      int sample_rate = 16000);

  /**
   * Assign speaker labels to ASR sentences based on diarization results
   * @param sentence_list List of sentences with timestamps
   * @param speaker_segments Speaker diarization results
   */
  void AssignSpeakersToSentences(
      std::vector<std::map<std::string, std::string>>& sentence_list,
      const std::vector<SpeakerSegment>& speaker_segments);

  // Getters
  int GetMinNumSpeakers() const { return min_num_speakers_; }
  int GetMaxNumSpeakers() const { return max_num_speakers_; }
  void SetMinNumSpeakers(int min_spk) { min_num_speakers_ = min_spk; }
  void SetMaxNumSpeakers(int max_spk) { max_num_speakers_ = max_spk; }

 private:
  // campplus_model_ is a non-owning pointer - the caller retains ownership
  // and must ensure the model outlives this instance
  CAMPPlusModel* campplus_model_ = nullptr;

  // clusterer_ is owned by this class - use unique_ptr for automatic memory
  // management
  std::unique_ptr<SpectralClustering> clusterer_;

  // Configuration
  float segment_duration_ = DEFAULT_SEGMENT_DURATION;
  float segment_shift_ = DEFAULT_SEGMENT_SHIFT;
  int min_num_speakers_ = DEFAULT_MIN_NUM_SPEAKERS;
  int max_num_speakers_ = DEFAULT_MAX_NUM_SPEAKERS;
  float merge_threshold_ = DEFAULT_MERGE_THRESHOLD;
  float min_segment_duration_ = DEFAULT_MIN_SEGMENT_DURATION;

  bool initialized_ = false;

  /**
   * Chunk VAD segments into smaller pieces for embedding extraction
   * @param vad_segments Input VAD segments
   * @param sample_rate Audio sample rate
   * @return Chunked segments
   */
  std::vector<std::tuple<float, float, std::vector<float>>> ChunkSegments(
      const std::vector<std::tuple<float, float, std::vector<float>>>&
          vad_segments,
      int sample_rate);

  /**
   * Extract fbank features from audio
   * @param audio_data Raw audio samples
   * @param sample_rate Audio sample rate
   * @return Fbank features
   */
  std::vector<float> ExtractFbankFeatures(const std::vector<float>& audio_data,
                                          int sample_rate);

  /**
   * Extract speaker embeddings from segments
   * @param segments Audio segments
   * @return Speaker embeddings
   */
  std::vector<std::vector<float>> ExtractEmbeddings(
      const std::vector<std::tuple<float, float, std::vector<float>>>&
          segments);

  /**
   * Post-process clustering results to get final speaker segments
   * @param segments Original segments with timestamps
   * @param labels Cluster labels
   * @param embeddings Speaker embeddings
   * @return Final speaker segments
   */
  std::vector<SpeakerSegment> PostProcess(
      const std::vector<std::tuple<float, float, std::vector<float>>>& segments,
      const std::vector<int>& labels,
      const std::vector<std::vector<float>>& embeddings);

  /**
   * Merge consecutive segments with same speaker
   */
  std::vector<SpeakerSegment> MergeConsecutiveSegments(
      std::vector<SpeakerSegment>& segments);

  /**
   * Smooth results by assigning short segments to nearest speakers
   */
  std::vector<SpeakerSegment> SmoothResults(
      std::vector<SpeakerSegment>& segments);

  /**
   * Merge similar speakers by cosine similarity
   */
  std::vector<int> MergeByCosineSimilarity(
      std::vector<int>& labels,
      const std::vector<std::vector<float>>& embeddings);
};

/**
 * Factory function to create speaker diarization
 */
SpeakerDiarization* CreateSpeakerDiarization(
    CAMPPlusModel* campplus_model,
    const std::map<std::string, std::string>& config = {});

// Utility functions

/**
 * Compute cosine similarity between two vectors
 */
inline float CosineSimilarity(const std::vector<float>& a,
                              const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) {
    return 0.0f;
  }

  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
  return denom > 1e-6f ? dot / denom : 0.0f;
}

/**
 * Compute cosine similarity matrix
 */
std::vector<std::vector<float>> ComputeCosineSimilarityMatrix(
    const std::vector<std::vector<float>>& embeddings);

}  // namespace funasr

#endif  // SPEAKER_DIARIZATION_H
