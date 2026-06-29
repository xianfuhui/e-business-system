package com.example.system.controller;

import com.example.system.service.ColabService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Controller
public class ColabController {

    @Autowired
    private ColabService colabService;

    // Trang web
    @GetMapping("/")
    public String home() {
        return "index";
    }

    // API gọi từ JS
    @PostMapping("/api/run")
    @ResponseBody
    public String run(
            @RequestParam("file")
            MultipartFile file
    ) {
        try {
            return colabService
                    .uploadToPython(file);
        } catch (Exception e) {
            return "{\"error\":\""
                    + e.getMessage()
                    + "\"}";
        }
    }

    @PostMapping("/api/llm")
    @ResponseBody
    public String llm() {
        try {
            return colabService.getLLMInsight();
        } catch (Exception e) {
            return "{\"llm_analysis\":\"" + e.getMessage() + "\"}";
        }
    }

    @PostMapping("/api/chat")
    @ResponseBody
    public String chat(@RequestBody String requestBody) {
        try {
            return colabService.chatWithColab(requestBody);
        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    @PostMapping("/api/predict")
    @ResponseBody
    public String predict(@RequestBody java.util.Map<String, Object> body) {
        try {
            java.util.List<String> sequence =
                    (java.util.List<String>) body.get("sequence");

            return colabService.predictNext(sequence);

        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }
}