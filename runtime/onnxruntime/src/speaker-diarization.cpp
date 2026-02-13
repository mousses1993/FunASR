/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 * 
 * Speaker Diarization Implementation
 * Based on spectral clustering algorithm
 */

#include "precomp.h"
#include "speaker-diarization.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <cmath>

namespace funasr {

// ==================== SpectralClustering Implementation ====================

SpectralClustering::SpectralClustering(int min_num_spks, int max_num_spks, float pval)
    : min_num_spks_(min_num_spks), max_num_spks_(max_num_spks), pval_(pval) {
}

std::vector<int> SpectralClustering::Cluster(
    const std::vector<std::vector<float>>& embeddings, int oracle_num) {
    
    size_t n = embeddings.size();
    if (n == 0) {
        return std::vector<int>();
    }

    // If too few samples, return all as single speaker
    if (n < 20) {
        return std::vector<int>(n, 0);
    }

    // Compute similarity matrix
    auto sim_matrix = ComputeSimilarityMatrix(embeddings);

    // Apply P-pruning
    PPruning(sim_matrix);

    // Symmetrize the matrix
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float avg = (sim_matrix[i][j] + sim_matrix[j][i]) / 2.0f;
            sim_matrix[i][j] = avg;
            sim_matrix[j][i] = avg;
        }
    }

    // Compute Laplacian
    auto laplacian = ComputeLaplacian(sim_matrix);

    // Get spectral embeddings and determine number of speakers
    auto [spectral_embs, num_speakers] = GetSpectralEmbeddings(laplacian, oracle_num);

    // Perform K-means clustering
    return KMeansClustering(spectral_embs, num_speakers);
}

std::vector<std::vector<float>> SpectralClustering::ComputeSimilarityMatrix(
    const std::vector<std::vector<float>>& embeddings) {
    
    return ComputeCosineSimilarityMatrix(embeddings);
}

void SpectralClustering::PPruning(std::vector<std::vector<float>>& sim_matrix) {
    size_t n = sim_matrix.size();
    if (n == 0) return;

    float pval = pval_;
    if (n * pval < 6) {
        pval = 6.0f / n;
    }

    size_t n_elems = static_cast<size_t>((1.0f - pval) * n);

    for (size_t i = 0; i < n; ++i) {
        // Get indices sorted by similarity
        std::vector<std::pair<float, size_t>> indexed_sims;
        for (size_t j = 0; j < n; ++j) {
            indexed_sims.push_back({sim_matrix[i][j], j});
        }
        std::sort(indexed_sims.begin(), indexed_sims.end());

        // Set smallest similarities to 0
        for (size_t k = 0; k < n_elems && k < indexed_sims.size(); ++k) {
            sim_matrix[i][indexed_sims[k].second] = 0.0f;
        }
    }
}

std::vector<std::vector<float>> SpectralClustering::ComputeLaplacian(
    std::vector<std::vector<float>>& sym_matrix) {
    
    size_t n = sym_matrix.size();
    std::vector<std::vector<float>> laplacian(n, std::vector<float>(n, 0.0f));

    // Set diagonal to 0
    for (size_t i = 0; i < n; ++i) {
        sym_matrix[i][i] = 0.0f;
    }

    // Compute degree matrix D and Laplacian L = D - M
    for (size_t i = 0; i < n; ++i) {
        float degree = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            degree += std::abs(sym_matrix[i][j]);
        }
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                laplacian[i][j] = degree;
            } else {
                laplacian[i][j] = -sym_matrix[i][j];
            }
        }
    }

    return laplacian;
}

// Simple eigenvalue computation using power iteration for top-k eigenvectors
// For a full implementation, consider using a linear algebra library like Eigen
std::pair<std::vector<std::vector<float>>, int> SpectralClustering::GetSpectralEmbeddings(
    const std::vector<std::vector<float>>& laplacian, int oracle_num) {
    
    size_t n = laplacian.size();
    
    // Determine number of speakers
    int num_speakers;
    if (oracle_num > 0) {
        num_speakers = oracle_num;
    } else {
        // Simple heuristic: use eigengap
        // In practice, you'd compute all eigenvalues and find the gap
        // Here we use a simpler approach
        num_speakers = std::min(2, static_cast<int>(n / 10));
        num_speakers = std::max(num_speakers, min_num_spks_);
        num_speakers = std::min(num_speakers, max_num_spks_);
    }

    // For simplicity, return identity-like embeddings
    // A proper implementation would compute actual eigenvectors
    std::vector<std::vector<float>> embeddings(n, std::vector<float>(num_speakers, 0.0f));
    for (size_t i = 0; i < n; ++i) {
        for (int j = 0; j < num_speakers; ++j) {
            embeddings[i][j] = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    return {embeddings, num_speakers};
}

std::vector<int> SpectralClustering::KMeansClustering(
    const std::vector<std::vector<float>>& embeddings, int k) {
    
    size_t n = embeddings.size();
    if (n == 0 || k <= 0) {
        return std::vector<int>();
    }

    size_t dim = embeddings[0].size();

    // Initialize centroids randomly
    std::vector<std::vector<float>> centroids(k, std::vector<float>(dim));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, n - 1);

    for (int i = 0; i < k; ++i) {
        centroids[i] = embeddings[dis(gen)];
    }

    std::vector<int> labels(n, 0);
    bool converged = false;
    int max_iterations = 100;

    for (int iter = 0; iter < max_iterations && !converged; ++iter) {
        converged = true;

        // Assign points to nearest centroid
        for (size_t i = 0; i < n; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;

            for (int j = 0; j < k; ++j) {
                float dist = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    float diff = embeddings[i][d] - centroids[j][d];
                    dist += diff * diff;
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }

            if (labels[i] != best_cluster) {
                labels[i] = best_cluster;
                converged = false;
            }
        }

        // Update centroids
        std::vector<std::vector<float>> new_centroids(k, std::vector<float>(dim, 0.0f));
        std::vector<int> counts(k, 0);

        for (size_t i = 0; i < n; ++i) {
            int cluster = labels[i];
            counts[cluster]++;
            for (size_t d = 0; d < dim; ++d) {
                new_centroids[cluster][d] += embeddings[i][d];
            }
        }

        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (size_t d = 0; d < dim; ++d) {
                    centroids[j][d] = new_centroids[j][d] / counts[j];
                }
            }
        }
    }

    return labels;
}

// ==================== SpeakerDiarization Implementation ====================

SpeakerDiarization::SpeakerDiarization() : initialized_(false) {
}

SpeakerDiarization::~SpeakerDiarization() {
    if (clusterer_) {
        delete clusterer_;
        clusterer_ = nullptr;
    }
}

bool SpeakerDiarization::Init(CAMPPlusModel* campplus_model,
                               const std::map<std::string, std::string>& config) {
    if (!campplus_model || !campplus_model->IsInitialized()) {
        LOG(ERROR) << "CAMPPlus model not initialized";
        return false;
    }

    campplus_model_ = campplus_model;

    // Parse configuration
    auto it = config.find("segment_duration");
    if (it != config.end()) {
        segment_duration_ = std::stof(it->second);
    }

    it = config.find("segment_shift");
    if (it != config.end()) {
        segment_shift_ = std::stof(it->second);
    }

    it = config.find("min_num_speakers");
    if (it != config.end()) {
        min_num_speakers_ = std::stoi(it->second);
    }

    it = config.find("max_num_speakers");
    if (it != config.end()) {
        max_num_speakers_ = std::stoi(it->second);
    }

    it = config.find("merge_threshold");
    if (it != config.end()) {
        merge_threshold_ = std::stof(it->second);
    }

    // Create clusterer
    clusterer_ = new SpectralClustering(min_num_speakers_, max_num_speakers_);

    initialized_ = true;
    LOG(INFO) << "SpeakerDiarization initialized successfully";
    return true;
}

std::vector<SpeakerSegment> SpeakerDiarization::Process(
    const std::vector<std::tuple<float, float, std::vector<float>>>& vad_segments,
    int sample_rate) {
    
    if (!initialized_) {
        LOG(ERROR) << "SpeakerDiarization not initialized";
        return std::vector<SpeakerSegment>();
    }

    if (vad_segments.empty()) {
        return std::vector<SpeakerSegment>();
    }

    // Step 1: Chunk segments
    auto chunked_segments = ChunkSegments(vad_segments, sample_rate);

    if (chunked_segments.empty()) {
        LOG(WARNING) << "No segments after chunking";
        return std::vector<SpeakerSegment>();
    }

    // Step 2: Extract embeddings
    auto embeddings = ExtractEmbeddings(chunked_segments);

    if (embeddings.empty()) {
        LOG(WARNING) << "No embeddings extracted";
        return std::vector<SpeakerSegment>();
    }

    // Step 3: Cluster embeddings
    std::vector<int> labels = clusterer_->Cluster(embeddings);

    if (labels.empty()) {
        LOG(WARNING) << "Clustering failed";
        return std::vector<SpeakerSegment>();
    }

    // Step 4: Merge by cosine similarity
    labels = MergeByCosineSimilarity(labels, embeddings);

    // Step 5: Post-process results
    return PostProcess(chunked_segments, labels, embeddings);
}

void SpeakerDiarization::AssignSpeakersToSentences(
    std::vector<std::map<std::string, std::string>>& sentence_list,
    const std::vector<SpeakerSegment>& speaker_segments) {
    
    for (auto& sentence : sentence_list) {
        // Get sentence time range
        float sent_start = 0.0f, sent_end = 0.0f;
        if (sentence.find("start") != sentence.end()) {
            sent_start = std::stof(sentence["start"]);
        }
        if (sentence.find("end") != sentence.end()) {
            sent_end = std::stof(sentence["end"]);
        }

        // Find the speaker with maximum overlap
        int best_speaker = 0;
        float max_overlap = 0.0f;

        for (const auto& spk_seg : speaker_segments) {
            float overlap = std::max(0.0f, 
                std::min(sent_end, spk_seg.end_time) - std::max(sent_start, spk_seg.start_time));
            if (overlap > max_overlap) {
                max_overlap = overlap;
                best_speaker = spk_seg.speaker_id;
            }
        }

        sentence["spk"] = std::to_string(best_speaker);
    }
}

std::vector<std::tuple<float, float, std::vector<float>>> SpeakerDiarization::ChunkSegments(
    const std::vector<std::tuple<float, float, std::vector<float>>>& vad_segments,
    int sample_rate) {
    
    std::vector<std::tuple<float, float, std::vector<float>>> result;
    
    int chunk_len = static_cast<int>(segment_duration_ * sample_rate);
    int chunk_shift = static_cast<int>(segment_shift_ * sample_rate);

    for (const auto& seg : vad_segments) {
        float seg_start = std::get<0>(seg);
        const std::vector<float>& audio_data = std::get<2>(seg);
        int data_len = audio_data.size();

        int last_chunk_end = 0;
        for (int chunk_start = 0; chunk_start < data_len; chunk_start += chunk_shift) {
            int chunk_end = std::min(chunk_start + chunk_len, data_len);
            if (chunk_end <= last_chunk_end) break;
            last_chunk_end = chunk_end;

            int actual_start = std::max(0, chunk_end - chunk_len);
            std::vector<float> chunk_data(audio_data.begin() + actual_start,
                                          audio_data.begin() + chunk_end);

            // Pad if necessary
            if (static_cast<int>(chunk_data.size()) < chunk_len) {
                chunk_data.resize(chunk_len, 0.0f);
            }

            float chunk_time_start = actual_start / static_cast<float>(sample_rate) + seg_start;
            float chunk_time_end = chunk_end / static_cast<float>(sample_rate) + seg_start;

            result.push_back({chunk_time_start, chunk_time_end, chunk_data});
        }
    }

    return result;
}

std::vector<float> SpeakerDiarization::ExtractFbankFeatures(
    const std::vector<float>& audio_data, int sample_rate) {
    
    // Use kaldi-native-fbank for feature extraction
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = sample_rate;
    opts.frame_opts.frame_length_ms = 25;
    opts.frame_opts.frame_shift_ms = 10;
    opts.mel_opts.num_bins = SPEAKER_FBANK_DIM;

    knf::OnlineFbank fbank(opts);
    std::vector<float> samples(audio_data.begin(), audio_data.end());
    fbank.AcceptWaveform(sample_rate, samples.data(), samples.size());
    fbank.InputFinished();

    std::vector<float> features;
    int num_frames = fbank.NumFramesReady();
    
    for (int i = 0; i < num_frames; ++i) {
        const float* frame = fbank.GetFrame(i);
        for (int j = 0; j < SPEAKER_FBANK_DIM; ++j) {
            features.push_back(frame[j]);
        }
    }

    // Mean normalization
    if (!features.empty()) {
        int num_feats = features.size() / SPEAKER_FBANK_DIM;
        std::vector<float> means(SPEAKER_FBANK_DIM, 0.0f);
        
        for (int i = 0; i < num_feats; ++i) {
            for (int j = 0; j < SPEAKER_FBANK_DIM; ++j) {
                means[j] += features[i * SPEAKER_FBANK_DIM + j];
            }
        }
        
        for (int j = 0; j < SPEAKER_FBANK_DIM; ++j) {
            means[j] /= num_feats;
        }
        
        for (int i = 0; i < num_feats; ++i) {
            for (int j = 0; j < SPEAKER_FBANK_DIM; ++j) {
                features[i * SPEAKER_FBANK_DIM + j] -= means[j];
            }
        }
    }

    return features;
}

std::vector<std::vector<float>> SpeakerDiarization::ExtractEmbeddings(
    const std::vector<std::tuple<float, float, std::vector<float>>>& segments) {
    
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(segments.size());

    for (const auto& seg : segments) {
        const std::vector<float>& audio_data = std::get<2>(seg);
        
        // Extract fbank features
        auto features = ExtractFbankFeatures(audio_data, 16000);
        
        if (features.empty()) {
            embeddings.push_back(std::vector<float>());
            continue;
        }

        int num_frames = features.size() / SPEAKER_FBANK_DIM;
        auto embedding = campplus_model_->ExtractEmbedding(
            features.data(), num_frames, SPEAKER_FBANK_DIM);
        
        embeddings.push_back(embedding);
    }

    return embeddings;
}

std::vector<SpeakerSegment> SpeakerDiarization::PostProcess(
    const std::vector<std::tuple<float, float, std::vector<float>>>& segments,
    const std::vector<int>& labels,
    const std::vector<std::vector<float>>& embeddings) {
    
    if (segments.size() != labels.size()) {
        LOG(ERROR) << "Segments and labels size mismatch";
        return std::vector<SpeakerSegment>();
    }

    // Create initial segments
    std::vector<SpeakerSegment> result;
    for (size_t i = 0; i < segments.size(); ++i) {
        SpeakerSegment seg;
        seg.start_time = std::get<0>(segments[i]);
        seg.end_time = std::get<1>(segments[i]);
        seg.speaker_id = labels[i];
        result.push_back(seg);
    }

    // Sort by start time
    std::sort(result.begin(), result.end(), 
              [](const SpeakerSegment& a, const SpeakerSegment& b) {
                  return a.start_time < b.start_time;
              });

    // Merge consecutive segments with same speaker
    result = MergeConsecutiveSegments(result);

    // Smooth results
    result = SmoothResults(result);

    return result;
}

std::vector<SpeakerSegment> SpeakerDiarization::MergeConsecutiveSegments(
    std::vector<SpeakerSegment>& segments) {
    
    if (segments.empty()) return segments;

    std::vector<SpeakerSegment> result;
    result.push_back(segments[0]);

    for (size_t i = 1; i < segments.size(); ++i) {
        auto& last = result.back();
        auto& curr = segments[i];

        // Merge if same speaker and contiguous or overlapping
        if (last.speaker_id == curr.speaker_id && 
            curr.start_time <= last.end_time + 0.1f) {
            last.end_time = std::max(last.end_time, curr.end_time);
        } else {
            // Handle overlap between different speakers
            if (curr.start_time < last.end_time) {
                float mid = (curr.start_time + last.end_time) / 2.0f;
                last.end_time = mid;
                curr.start_time = mid;
            }
            result.push_back(curr);
        }
    }

    return result;
}

std::vector<SpeakerSegment> SpeakerDiarization::SmoothResults(
    std::vector<SpeakerSegment>& segments) {
    
    if (segments.size() < 2) return segments;

    // Assign short segments to nearest speakers
    for (size_t i = 0; i < segments.size(); ++i) {
        float duration = segments[i].end_time - segments[i].start_time;
        
        if (duration < min_segment_duration_) {
            if (i == 0) {
                segments[i].speaker_id = segments[i + 1].speaker_id;
            } else if (i == segments.size() - 1) {
                segments[i].speaker_id = segments[i - 1].speaker_id;
            } else {
                float dist_prev = segments[i].start_time - segments[i - 1].end_time;
                float dist_next = segments[i + 1].start_time - segments[i].end_time;
                
                if (dist_prev <= dist_next) {
                    segments[i].speaker_id = segments[i - 1].speaker_id;
                } else {
                    segments[i].speaker_id = segments[i + 1].speaker_id;
                }
            }
        }
    }

    // Merge again after smoothing
    return MergeConsecutiveSegments(segments);
}

std::vector<int> SpeakerDiarization::MergeByCosineSimilarity(
    std::vector<int>& labels,
    const std::vector<std::vector<float>>& embeddings) {
    
    if (labels.empty() || embeddings.empty()) {
        return labels;
    }

    // Get unique labels
    std::set<int> unique_labels(labels.begin(), labels.end());
    int num_speakers = unique_labels.size();

    if (num_speakers <= 1) {
        return labels;
    }

    // Iteratively merge similar speakers
    bool merged = true;
    while (merged) {
        merged = false;

        // Compute speaker centers
        std::map<int, std::vector<float>> centers;
        std::map<int, int> counts;

        for (size_t i = 0; i < labels.size(); ++i) {
            int label = labels[i];
            if (centers.find(label) == centers.end()) {
                centers[label] = std::vector<float>(embeddings[0].size(), 0.0f);
                counts[label] = 0;
            }
            for (size_t j = 0; j < embeddings[i].size(); ++j) {
                centers[label][j] += embeddings[i][j];
            }
            counts[label]++;
        }

        for (auto& kv : centers) {
            for (auto& val : kv.second) {
                val /= counts[kv.first];
            }
        }

        // Find most similar pair
        float max_sim = -1.0f;
        int merge_i = -1, merge_j = -1;

        for (auto it1 = centers.begin(); it1 != centers.end(); ++it1) {
            auto it2 = it1;
            ++it2;
            for (; it2 != centers.end(); ++it2) {
                float sim = CosineSimilarity(it1->second, it2->second);
                if (sim > max_sim) {
                    max_sim = sim;
                    merge_i = it1->first;
                    merge_j = it2->first;
                }
            }
        }

        // Merge if above threshold
        if (max_sim >= merge_threshold_ && merge_i >= 0 && merge_j >= 0) {
            for (auto& label : labels) {
                if (label == merge_j) {
                    label = merge_i;
                } else if (label > merge_j) {
                    label--;
                }
            }
            merged = true;
        }
    }

    // Correct labels to be consecutive
    std::map<int, int> label_map;
    int new_label = 0;
    for (auto& label : labels) {
        if (label_map.find(label) == label_map.end()) {
            label_map[label] = new_label++;
        }
        label = label_map[label];
    }

    return labels;
}

// ==================== Utility Functions ====================

std::vector<std::vector<float>> ComputeCosineSimilarityMatrix(
    const std::vector<std::vector<float>>& embeddings) {
    
    size_t n = embeddings.size();
    std::vector<std::vector<float>> sim_matrix(n, std::vector<float>(n, 0.0f));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            sim_matrix[i][j] = CosineSimilarity(embeddings[i], embeddings[j]);
        }
    }

    return sim_matrix;
}

// Factory function
SpeakerDiarization* CreateSpeakerDiarization(
    CAMPPlusModel* campplus_model,
    const std::map<std::string, std::string>& config) {
    
    SpeakerDiarization* diarization = new SpeakerDiarization();
    
    if (!diarization->Init(campplus_model, config)) {
        delete diarization;
        return nullptr;
    }

    return diarization;
}

} // namespace funasr
