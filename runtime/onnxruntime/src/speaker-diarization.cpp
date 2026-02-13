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
    if (n == 1) {
        return std::vector<int>(1, 0);
    }

    // Use agglomerative clustering based on cosine similarity.
    // This is deterministic and avoids the eigenvector computation issues
    // with power iteration on the Laplacian.

    // Compute cosine similarity matrix
    auto sim_matrix = ComputeSimilarityMatrix(embeddings);

    // Initialize: each segment is its own cluster
    std::vector<int> labels(n);
    std::iota(labels.begin(), labels.end(), 0);
    int num_clusters = static_cast<int>(n);

    // Determine stopping criteria
    int target_clusters = oracle_num > 0 ? oracle_num : -1;
    // Similarity threshold for auto mode: use pval_ remapped to [0.5, 0.9]
    float stop_threshold = 0.5f;

    // Agglomerative clustering with average linkage
    while (num_clusters > 1) {
        if (target_clusters > 0 && num_clusters <= target_clusters) break;
        if (num_clusters <= min_num_spks_) break;

        // Compute average similarity between each pair of clusters
        float best_sim = -1.0f;
        int best_a = -1, best_b = -1;

        // Build cluster membership
        std::map<int, std::vector<size_t>> cluster_members;
        for (size_t i = 0; i < n; ++i) {
            cluster_members[labels[i]].push_back(i);
        }

        // Find the most similar pair of clusters (average linkage)
        auto it1 = cluster_members.begin();
        for (; it1 != cluster_members.end(); ++it1) {
            auto it2 = it1;
            ++it2;
            for (; it2 != cluster_members.end(); ++it2) {
                float total_sim = 0.0f;
                int count = 0;
                for (size_t i : it1->second) {
                    for (size_t j : it2->second) {
                        total_sim += sim_matrix[i][j];
                        count++;
                    }
                }
                float avg_sim = count > 0 ? total_sim / count : 0.0f;
                if (avg_sim > best_sim) {
                    best_sim = avg_sim;
                    best_a = it1->first;
                    best_b = it2->first;
                }
            }
        }

        // In auto mode, stop if best similarity is below threshold
        if (target_clusters < 0 && best_sim < stop_threshold) break;

        // Merge cluster best_b into best_a
        if (best_a >= 0 && best_b >= 0) {
            for (auto& l : labels) {
                if (l == best_b) l = best_a;
            }
            num_clusters--;
        } else {
            break;
        }
    }

    // Renumber labels to be consecutive starting from 0
    std::map<int, int> label_map;
    int new_label = 0;
    for (auto& l : labels) {
        if (label_map.find(l) == label_map.end()) {
            label_map[l] = new_label++;
        }
        l = label_map[l];
    }

    return labels;
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

// Power iteration to compute the dominant eigenvector of a matrix
static std::vector<float> PowerIteration(
    const std::vector<std::vector<float>>& matrix,
    int max_iterations = 100,
    float tolerance = 1e-6f) {
    
    size_t n = matrix.size();
    if (n == 0) return std::vector<float>();
    
    // Initialize with random values using mt19937 for thread safety
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = dis(gen);
    }
    
    // Normalize initial vector
    float norm = 0.0f;
    for (float val : v) norm += val * val;
    norm = std::sqrt(norm);
    if (norm > 1e-10f) {
        for (float& val : v) val /= norm;
    }
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Matrix-vector multiplication: v_new = matrix * v
        std::vector<float> v_new(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                v_new[i] += matrix[i][j] * v[j];
            }
        }
        
        // Normalize
        float new_norm = 0.0f;
        for (float val : v_new) new_norm += val * val;
        new_norm = std::sqrt(new_norm);
        if (new_norm < 1e-10f) break;
        
        for (float& val : v_new) val /= new_norm;
        
        // Check convergence
        float diff = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            diff += std::abs(v_new[i] - v[i]);
        }
        
        v = v_new;
        
        if (diff < tolerance) break;
    }
    
    return v;
}

// Compute top-k eigenvectors using deflation method
static std::vector<std::vector<float>> ComputeTopKEigenvectors(
    std::vector<std::vector<float>> matrix, int k, int max_iterations = 100) {
    
    size_t n = matrix.size();
    std::vector<std::vector<float>> eigenvectors(k, std::vector<float>(n, 0.0f));
    
    for (int eig_idx = 0; eig_idx < k; ++eig_idx) {
        // Power iteration for current eigenvector
        auto v = PowerIteration(matrix, max_iterations);
        if (v.empty()) continue;
        
        eigenvectors[eig_idx] = v;
        
        // Deflation: subtract the outer product of v from matrix
        // This removes the contribution of the found eigenvector
        // A' = A - lambda * v * v^T (approximately)
        // Compute eigenvalue estimate: lambda = v^T * A * v
        float eigenvalue = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                eigenvalue += v[i] * matrix[i][j] * v[j];
            }
        }
        
        // Deflate the matrix
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                matrix[i][j] -= eigenvalue * v[i] * v[j];
            }
        }
    }
    
    return eigenvectors;
}

// Compute eigenvalues for eigengap analysis
static std::vector<float> ComputeEigenvalues(
    const std::vector<std::vector<float>>& matrix, int num_eigenvalues) {
    
    std::vector<float> eigenvalues;
    std::vector<std::vector<float>> matrix_copy = matrix;
    
    for (int i = 0; i < num_eigenvalues; ++i) {
        auto v = PowerIteration(matrix_copy);
        if (v.empty()) break;
        
        // Compute eigenvalue: lambda = v^T * A * v
        float eigenvalue = 0.0f;
        size_t n = matrix_copy.size();
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                eigenvalue += v[j] * matrix_copy[j][k] * v[k];
            }
        }
        eigenvalues.push_back(eigenvalue);
        
        // Deflate for next eigenvalue
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                matrix_copy[j][k] -= eigenvalue * v[j] * v[k];
            }
        }
    }
    
    return eigenvalues;
}

// Use eigengap heuristic to determine optimal number of clusters
static int EstimateNumSpeakersByEigengap(const std::vector<float>& eigenvalues, 
                                          int min_spks, int max_spks) {
    if (eigenvalues.size() < 2) return min_spks;
    
    // Sort eigenvalues in descending order (they should already be sorted by power iteration)
    // Find the largest gap between consecutive eigenvalues
    float max_gap = 0.0f;
    int optimal_k = min_spks;
    
    int check_limit = std::min(static_cast<int>(eigenvalues.size()), max_spks + 1);
    
    for (int i = 1; i < check_limit; ++i) {
        float gap = eigenvalues[i - 1] - eigenvalues[i];
        if (gap > max_gap && i >= min_spks) {
            max_gap = gap;
            optimal_k = i;
        }
    }
    
    return std::max(min_spks, std::min(optimal_k, max_spks));
}

// Compute spectral embeddings using power iteration for eigenvector computation
std::pair<std::vector<std::vector<float>>, int> SpectralClustering::GetSpectralEmbeddings(
    const std::vector<std::vector<float>>& laplacian, int oracle_num) {
    
    size_t n = laplacian.size();
    
    // Determine number of speakers
    int num_speakers;
    if (oracle_num > 0) {
        num_speakers = oracle_num;
    } else {
        // Use eigengap heuristic to estimate number of speakers
        // Compute more eigenvalues than max_num_spks_ to find the gap
        int num_eigenvalues_to_compute = std::min(static_cast<int>(n), max_num_spks_ + 3);
        auto eigenvalues = ComputeEigenvalues(laplacian, num_eigenvalues_to_compute);
        
        if (eigenvalues.size() >= 2) {
            num_speakers = EstimateNumSpeakersByEigengap(eigenvalues, min_num_spks_, max_num_spks_);
        } else {
            // Fallback to simple heuristic
            num_speakers = std::max(2, static_cast<int>(n / 10));
            num_speakers = std::max(num_speakers, min_num_spks_);
            num_speakers = std::min(num_speakers, max_num_spks_);
        }
    }
    
    // Compute top-k eigenvectors using power iteration with deflation
    auto eigenvectors = ComputeTopKEigenvectors(laplacian, num_speakers);
    
    // Transpose to get embeddings: each row is a sample's embedding
    std::vector<std::vector<float>> embeddings(n, std::vector<float>(num_speakers, 0.0f));
    for (size_t i = 0; i < n; ++i) {
        for (int j = 0; j < num_speakers; ++j) {
            embeddings[i][j] = eigenvectors[j][i];
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

    // KMeans++ initialization for better convergence
    std::vector<std::vector<float>> centroids(k, std::vector<float>(dim));
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Step 1: Choose first centroid randomly
    std::uniform_int_distribution<size_t> dis(0, n - 1);
    centroids[0] = embeddings[dis(gen)];
    
    // Step 2: Choose remaining centroids with probability proportional to distance squared
    for (int c = 1; c < k; ++c) {
        std::vector<float> distances(n);
        float total_dist = 0.0f;
        
        for (size_t i = 0; i < n; ++i) {
            // Find distance to nearest existing centroid
            float min_dist = std::numeric_limits<float>::max();
            for (int j = 0; j < c; ++j) {
                float dist = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    float diff = embeddings[i][d] - centroids[j][d];
                    dist += diff * diff;
                }
                min_dist = std::min(min_dist, dist);
            }
            distances[i] = min_dist;
            total_dist += min_dist;
        }
        
        // Choose next centroid with weighted probability
        std::uniform_real_distribution<float> prob_dis(0.0f, total_dist);
        float threshold = prob_dis(gen);
        float cum_dist = 0.0f;
        size_t chosen = 0;
        
        for (size_t i = 0; i < n; ++i) {
            cum_dist += distances[i];
            if (cum_dist >= threshold) {
                chosen = i;
                break;
            }
        }
        
        centroids[c] = embeddings[chosen];
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
    // clusterer_ is a unique_ptr, so it's automatically cleaned up
    // No need for manual delete
}

bool SpeakerDiarization::Init(CAMPPlusModel* campplus_model,
                               const std::map<std::string, std::string>& config) {
    if (!campplus_model || !campplus_model->IsInitialized()) {
        LOG(ERROR) << "CAMPPlus model not initialized";
        return false;
    }

    campplus_model_ = campplus_model;

    // Parse configuration with exception handling
    try {
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
    } catch (const std::invalid_argument& e) {
        LOG(ERROR) << "Invalid configuration value: " << e.what();
        return false;
    } catch (const std::out_of_range& e) {
        LOG(ERROR) << "Configuration value out of range: " << e.what();
        return false;
    }

    // Validate configuration values
    if (segment_duration_ <= 0.0f) {
        LOG(ERROR) << "Invalid segment_duration: " << segment_duration_ << ", using default";
        segment_duration_ = DEFAULT_SEGMENT_DURATION;
    }
    if (segment_shift_ <= 0.0f) {
        LOG(ERROR) << "Invalid segment_shift: " << segment_shift_ << ", using default";
        segment_shift_ = DEFAULT_SEGMENT_SHIFT;
    }
    if (min_num_speakers_ < 1) {
        LOG(ERROR) << "Invalid min_num_speakers: " << min_num_speakers_ << ", using default";
        min_num_speakers_ = DEFAULT_MIN_NUM_SPEAKERS;
    }
    if (max_num_speakers_ < min_num_speakers_) {
        LOG(ERROR) << "max_num_speakers (" << max_num_speakers_ 
                   << ") < min_num_speakers (" << min_num_speakers_ << "), adjusting";
        max_num_speakers_ = min_num_speakers_;
    }
    if (merge_threshold_ < 0.0f || merge_threshold_ > 1.0f) {
        LOG(ERROR) << "Invalid merge_threshold: " << merge_threshold_ << ", using default";
        merge_threshold_ = DEFAULT_MERGE_THRESHOLD;
    }

    // Create clusterer using unique_ptr for automatic memory management
    clusterer_ = std::make_unique<SpectralClustering>(min_num_speakers_, max_num_speakers_);

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
            // Only remap labels: merge j -> i, do NOT decrement any labels
            // The label renumbering will be done at the end
            for (auto& label : labels) {
                if (label == merge_j) {
                    label = merge_i;
                }
            }
            merged = true;
        }
    }

    // Renumber labels to be consecutive starting from 0
    std::map<int, int> label_map;
    int new_label = 0;
    for (const int& original_label : unique_labels) {
        // Check if this label still exists after merging
        bool exists = false;
        for (int l : labels) {
            if (l == original_label) {
                exists = true;
                break;
            }
        }
        if (exists && label_map.find(original_label) == label_map.end()) {
            label_map[original_label] = new_label++;
        }
    }
    
    // Apply the final mapping
    for (auto& label : labels) {
        label = label_map[label];
    }

    return labels;
}

// ==================== Utility Functions ====================

std::vector<std::vector<float>> ComputeCosineSimilarityMatrix(
    const std::vector<std::vector<float>>& embeddings) {
    
    size_t n = embeddings.size();
    std::vector<std::vector<float>> sim_matrix(n, std::vector<float>(n, 0.0f));

    // Only compute upper triangle (including diagonal), then mirror
    // Cosine similarity is symmetric: sim[i][j] = sim[j][i]
    for (size_t i = 0; i < n; ++i) {
        sim_matrix[i][i] = 1.0f;  // Self-similarity is always 1.0
        for (size_t j = i + 1; j < n; ++j) {
            float sim = CosineSimilarity(embeddings[i], embeddings[j]);
            sim_matrix[i][j] = sim;
            sim_matrix[j][i] = sim;  // Mirror for symmetry
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
