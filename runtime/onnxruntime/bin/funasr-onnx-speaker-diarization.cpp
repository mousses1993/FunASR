/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 * 
 * Speaker Diarization Test Program
 * Demonstrates how to use the speaker diarization functionality
 */

#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <glog/logging.h>
#include "funasrruntime.h"
#include "campplus-model.h"
#include "speaker-diarization.h"
#include "tclap/CmdLine.h"
#include "com-define.h"
#include "util.h"
#include "audio.h"

using namespace std;

bool is_target_file(const std::string& filename, const std::string target) {
    std::size_t pos = filename.find_last_of(".");
    if (pos == std::string::npos) {
        return false;
    }
    std::string extension = filename.substr(pos + 1);
    return (extension == target);
}

void GetValue(TCLAP::ValueArg<std::string>& value_arg, string key, std::map<std::string, std::string>& model_path)
{
    model_path.insert({key, value_arg.getValue()});
    LOG(INFO)<< key << " : " << value_arg.getValue();
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-onnx-speaker-diarization", ' ', "1.0");
    
    // Speaker model parameters
    TCLAP::ValueArg<std::string>    speaker_dir("", SPEAKER_DIR, "the speaker model path (CAM++), which contains model.onnx", true, "", "string");
    TCLAP::ValueArg<std::string>    speaker_quant("", SPEAKER_QUANT, "true (Default), load the model of model.onnx in speaker_dir. If set true, load the model of model_quant.onnx", false, "true", "string");
    
    // VAD model parameters (optional, for segmenting audio)
    TCLAP::ValueArg<std::string>    vad_dir("", VAD_DIR, "the vad model path, which contains model.onnx, vad.yaml, vad.mvn", false, "", "string");
    TCLAP::ValueArg<std::string>    vad_quant("", VAD_QUANT, "true (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir", false, "true", "string");

    // Input/Output parameters
    TCLAP::ValueArg<std::string>    wav_path("", WAV_PATH, "the input audio file (wav, pcm, or other formats supported by ffmpeg)", true, "", "string");
    TCLAP::ValueArg<std::int32_t>   audio_fs("", AUDIO_FS, "the sample rate of audio", false, 16000, "int32_t");
    TCLAP::ValueArg<std::string>    output_file("", "output", "output file path for diarization results", false, "", "string");
    
    // Speaker diarization parameters
    TCLAP::ValueArg<std::int32_t>   min_speakers("", "min-speakers", "minimum number of speakers", false, 1, "int32_t");
    TCLAP::ValueArg<std::int32_t>   max_speakers("", "max-speakers", "maximum number of speakers", false, 15, "int32_t");

    cmd.add(speaker_dir);
    cmd.add(speaker_quant);
    cmd.add(vad_dir);
    cmd.add(vad_quant);
    cmd.add(wav_path);
    cmd.add(audio_fs);
    cmd.add(output_file);
    cmd.add(min_speakers);
    cmd.add(max_speakers);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(speaker_dir, SPEAKER_DIR, model_path);
    GetValue(speaker_quant, SPEAKER_QUANT, model_path);
    GetValue(wav_path, WAV_PATH, model_path);

    struct timeval start, end;
    gettimeofday(&start, nullptr);
    int thread_num = 1;

    // Initialize CAM++ speaker model
    LOG(INFO) << "Initializing CAM++ speaker model...";
    FUNASR_HANDLE campplus_handle = CampplusInit(model_path, thread_num);
    if (!campplus_handle) {
        LOG(ERROR) << "Failed to initialize CAM++ speaker model";
        exit(-1);
    }
    gettimeofday(&end, nullptr);
    long seconds = (end.tv_sec - start.tv_sec);
    long model_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    LOG(INFO) << "CAM++ model initialization takes " << (double)model_init_micros / 1000000 << " s";

    // Initialize VAD if provided
    FUNASR_HANDLE vad_handle = nullptr;
    if (vad_dir.isSet()) {
        gettimeofday(&start, nullptr);
        LOG(INFO) << "Initializing VAD model from: " << vad_dir.getValue();
        // FsmnVadInit expects MODEL_DIR and QUANTIZE keys (not VAD_DIR/VAD_QUANT)
        std::map<std::string, std::string> vad_model_path;
        vad_model_path[MODEL_DIR] = vad_dir.getValue();
        vad_model_path[QUANTIZE] = vad_quant.getValue();
        vad_handle = FsmnVadInit(vad_model_path, thread_num);
        if (!vad_handle) {
            LOG(ERROR) << "Failed to initialize VAD model";
            CampplusUninit(campplus_handle);
            exit(-1);
        }
        gettimeofday(&end, nullptr);
        seconds = (end.tv_sec - start.tv_sec);
        model_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
        LOG(INFO) << "VAD model initialization takes " << (double)model_init_micros / 1000000 << " s";
    }

    // Initialize speaker diarization
    std::map<std::string, std::string> diar_config;
    diar_config["min_num_speakers"] = std::to_string(min_speakers.getValue());
    diar_config["max_num_speakers"] = std::to_string(max_speakers.getValue());
    
    LOG(INFO) << "Initializing speaker diarization...";
    FUNASR_HANDLE diar_handle = SpeakerDiarizationInit(campplus_handle, diar_config);
    if (!diar_handle) {
        LOG(ERROR) << "Failed to initialize speaker diarization";
        if (vad_handle) FsmnVadUninit(vad_handle);
        CampplusUninit(campplus_handle);
        exit(-1);
    }

    // Load audio
    string wav_file = model_path.at(WAV_PATH);
    int32_t sampling_rate = audio_fs.getValue();
    
    LOG(INFO) << "Loading audio file: " << wav_file;
    funasr::Audio audio(1);
    
    if (is_target_file(wav_file.c_str(), "wav")) {
        if (!audio.LoadWav2Char(wav_file.c_str(), &sampling_rate)) {
            LOG(ERROR) << "Failed to load " << wav_file;
            exit(-1);
        }
    } else if (is_target_file(wav_file.c_str(), "pcm")) {
        if (!audio.LoadPcmwav2Char(wav_file.c_str(), &sampling_rate)) {
            LOG(ERROR) << "Failed to load " << wav_file;
            exit(-1);
        }
    } else {
        if (!audio.FfmpegLoad(wav_file.c_str(), true)) {
            LOG(ERROR) << "Failed to load " << wav_file;
            exit(-1);
        }
    }

    char* speech_buff = audio.GetSpeechChar();
    int buff_len = audio.GetSpeechLen() * 2;
    float audio_duration = audio.GetTimeLen();

    LOG(INFO) << "Audio duration: " << audio_duration << " seconds";

    // Run VAD if available
    std::vector<std::tuple<float, float, std::vector<float>>> vad_segments;
    
    if (vad_handle) {
        LOG(INFO) << "Running VAD...";
        gettimeofday(&start, nullptr);
        
        FUNASR_RESULT vad_result = FsmnVadInferBuffer(vad_handle, speech_buff, buff_len, nullptr, true, sampling_rate, "pcm");
        
        gettimeofday(&end, nullptr);
        seconds = (end.tv_sec - start.tv_sec);
        long vad_time = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
        LOG(INFO) << "VAD takes " << (double)vad_time / 1000000 << " s";

        if (vad_result) {
            std::vector<std::vector<int>>* vad_res = FsmnVadGetResult(vad_result, 0);
            if (vad_res && !vad_res->empty()) {
                LOG(INFO) << "VAD detected " << vad_res->size() << " speech segments";
                
                // Convert VAD result to audio segments
                for (const auto& seg : *vad_res) {
                    if (seg.size() >= 2) {
                        int start_sample = seg[0];
                        int end_sample = seg[1];
                        
                        float start_time = start_sample / 1000.0f;  // VAD outputs ms
                        float end_time = end_sample / 1000.0f;
                        
                        // Extract audio data for this segment
                        int start_idx = static_cast<int>(start_time * sampling_rate) * 2;
                        int end_idx = static_cast<int>(end_time * sampling_rate) * 2;
                        
                        start_idx = std::max(0, start_idx);
                        end_idx = std::min(buff_len, end_idx);
                        
                        std::vector<float> segment_audio;
                        segment_audio.reserve((end_idx - start_idx) / 2);
                        
                        for (int i = start_idx; i < end_idx; i += 2) {
                            int16_t sample = *reinterpret_cast<int16_t*>(speech_buff + i);
                            segment_audio.push_back(static_cast<float>(sample) / 32768.0f);
                        }
                        
                        vad_segments.push_back({start_time, end_time, segment_audio});
                    }
                }
            }
            FsmnVadFreeResult(vad_result);
        }
    } else {
        // If no VAD, use the entire audio as one segment
        LOG(INFO) << "No VAD model, using entire audio as one segment";
        std::vector<float> full_audio;
        full_audio.reserve(buff_len / 2);
        for (int i = 0; i < buff_len; i += 2) {
            int16_t sample = *reinterpret_cast<int16_t*>(speech_buff + i);
            full_audio.push_back(static_cast<float>(sample) / 32768.0f);
        }
        vad_segments.push_back({0.0f, audio_duration, full_audio});
    }

    // Run speaker diarization
    if (!vad_segments.empty()) {
        LOG(INFO) << "Running speaker diarization on " << vad_segments.size() << " segments...";
        gettimeofday(&start, nullptr);
        
        const char* diar_result = SpeakerDiarizationProcess(diar_handle, vad_segments, sampling_rate);
        
        gettimeofday(&end, nullptr);
        seconds = (end.tv_sec - start.tv_sec);
        long diar_time = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
        LOG(INFO) << "Speaker diarization takes " << (double)diar_time / 1000000 << " s";

        if (diar_result) {
            LOG(INFO) << "Speaker diarization result: " << diar_result;
            
            // Save to file if specified
            if (output_file.isSet()) {
                std::ofstream out_file(output_file.getValue());
                if (out_file.is_open()) {
                    out_file << "{\n";
                    out_file << "  \"audio_file\": \"" << wav_file << "\",\n";
                    out_file << "  \"duration\": " << audio_duration << ",\n";
                    out_file << "  \"segments\": " << diar_result << "\n";
                    out_file << "}\n";
                    out_file.close();
                    LOG(INFO) << "Results saved to: " << output_file.getValue();
                }
            }
            
            SpeakerDiarizationFreeResult(diar_result);
        } else {
            LOG(ERROR) << "Speaker diarization failed";
        }
    } else {
        LOG(WARNING) << "No speech segments detected";
    }

    // Cleanup
    LOG(INFO) << "Cleaning up...";
    SpeakerDiarizationUninit(diar_handle);
    if (vad_handle) FsmnVadUninit(vad_handle);
    CampplusUninit(campplus_handle);

    LOG(INFO) << "Done!";
    return 0;
}
