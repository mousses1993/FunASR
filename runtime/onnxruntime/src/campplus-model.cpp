/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 * 
 * CAMPPlus Model Implementation
 * Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)
 */

#include "precomp.h"
#include "campplus-model.h"
#include <algorithm>
#include <cmath>

namespace funasr {

CAMPPlusModel::CAMPPlusModel() : initialized_(false) {
}

CAMPPlusModel::~CAMPPlusModel() {
    session_.reset();
}

bool CAMPPlusModel::InitModel(const std::string& model_path,
                               const std::string& config_path,
                               int thread_num) {
    if (initialized_) {
        LOG(WARNING) << "CAMPPlusModel already initialized";
        return true;
    }

    try {
        // Set session options
        session_options_.SetIntraOpNumThreads(thread_num);
        session_options_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Create ONNX session
        session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(), 
                                                   session_options_);

        // Get input names
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_->GetInputCount();
        input_names_.resize(num_inputs);
        input_name_ptrs_.resize(num_inputs);

        for (size_t i = 0; i < num_inputs; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_[i] = input_name.get();
            input_name_ptrs_[i] = input_names_[i].c_str();
        }

        // Get output names
        size_t num_outputs = session_->GetOutputCount();
        output_names_.resize(num_outputs);
        output_name_ptrs_.resize(num_outputs);

        for (size_t i = 0; i < num_outputs; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_[i] = output_name.get();
            output_name_ptrs_[i] = output_names_[i].c_str();
        }

        thread_num_ = thread_num;
        initialized_ = true;

        LOG(INFO) << "CAMPPlusModel initialized successfully from: " << model_path;
        LOG(INFO) << "Input names: " << input_names_.size() 
                  << ", Output names: " << output_names_.size();

        return true;
    } catch (const Ort::Exception& e) {
        LOG(ERROR) << "ONNX Runtime error during CAMPPlusModel initialization: " << e.what();
        return false;
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error initializing CAMPPlusModel: " << e.what();
        return false;
    }
}

std::vector<float> CAMPPlusModel::ExtractEmbedding(const float* features,
                                                    int num_frames,
                                                    int feat_dim) {
    if (!initialized_) {
        LOG(ERROR) << "CAMPPlusModel not initialized";
        return std::vector<float>();
    }

    try {
        // Create input tensor
        // Input shape: [batch_size, num_frames, feat_dim]
        std::vector<int64_t> input_shape = {1, num_frames, feat_dim};
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<float> input_data(features, features + num_frames * feat_dim);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs_.data(), &input_tensor, 1,
            output_name_ptrs_.data(), output_name_ptrs_.size());

        // Get output embedding
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t embedding_size = 1;
        for (auto dim : output_shape) {
            embedding_size *= dim;
        }

        std::vector<float> embedding(output_data, output_data + embedding_size);

        // L2 normalize the embedding
        float norm = 0.0f;
        for (float val : embedding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm > 1e-6f) {
            for (float& val : embedding) {
                val /= norm;
            }
        }

        return embedding;
    } catch (const Ort::Exception& e) {
        LOG(ERROR) << "ONNX Runtime error during embedding extraction: " << e.what();
        return std::vector<float>();
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error extracting embedding: " << e.what();
        return std::vector<float>();
    }
}

std::vector<std::vector<float>> CAMPPlusModel::ExtractEmbeddings(
    const std::vector<std::tuple<float, float, std::vector<float>>>& audio_segments) {
    
    // DEPRECATED: This method is not used by SpeakerDiarization.
    // SpeakerDiarization::ExtractEmbeddings extracts fbank features internally
    // and calls ExtractEmbedding() directly.
    //
    // This method expects pre-computed fbank features (not raw audio) as input.
    // If you need to extract embeddings from raw audio, use the following pattern:
    //   1. Call SpeakerDiarization::ExtractFbankFeatures() to get fbank features
    //   2. Call CAMPPlusModel::ExtractEmbedding() with the fbank features
    //
    // This method is kept for API compatibility but may be removed in future versions.
    
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(audio_segments.size());

    for (const auto& segment : audio_segments) {
        const std::vector<float>& audio_data = std::get<2>(segment);
        
        // Input validation: expect pre-computed fbank features
        // Features should be in shape [num_frames * SPEAKER_FBANK_DIM]
        if (audio_data.size() % SPEAKER_FBANK_DIM != 0) {
            LOG(WARNING) << "Audio data size not divisible by feature dimension. "
                         << "Expected pre-computed fbank features, got size: " << audio_data.size();
            embeddings.push_back(std::vector<float>());
            continue;
        }

        int num_frames = audio_data.size() / SPEAKER_FBANK_DIM;
        if (num_frames == 0) {
            LOG(WARNING) << "Empty features provided";
            embeddings.push_back(std::vector<float>());
            continue;
        }
        
        auto embedding = ExtractEmbedding(audio_data.data(), num_frames, SPEAKER_FBANK_DIM);
        embeddings.push_back(embedding);
    }

    return embeddings;
}

CAMPPlusModel* CreateCAMPPlusModel(const std::map<std::string, std::string>& model_path,
                                   int thread_num) {
    CAMPPlusModel* model = new CAMPPlusModel();
    
    std::string model_file;
    std::string config_file;

    // Get model path
    auto it = model_path.find(SPEAKER_DIR);
    if (it != model_path.end()) {
        model_file = PathAppend(it->second, MODEL_NAME);
        
        // Check for quantized model
        auto quant_it = model_path.find(SPEAKER_QUANT);
        if (quant_it != model_path.end() && quant_it->second == "true") {
            model_file = PathAppend(it->second, QUANT_MODEL_NAME);
        }
        
        config_file = PathAppend(it->second, CAMPP_CONFIG_NAME);
    }

    if (model_file.empty()) {
        LOG(ERROR) << "Model path not specified";
        delete model;
        return nullptr;
    }

    if (!model->InitModel(model_file, config_file, thread_num)) {
        delete model;
        return nullptr;
    }

    return model;
}

} // namespace funasr
