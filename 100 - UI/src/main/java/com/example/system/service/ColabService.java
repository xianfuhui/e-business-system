package com.example.system.service;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import org.springframework.beans.factory.annotation.Value;

@Service
public class ColabService {
    @Value("${python.api.base-url}")
    private String baseUrl;

    public String getLLMInsight() throws Exception {

        String url = baseUrl + "/api/llm";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<String> request = new HttpEntity<>("{}", headers);

        RestTemplate restTemplate = new RestTemplate();

        ResponseEntity<String> response =
                restTemplate.postForEntity(url, request, String.class);

        return response.getBody();
    }

    public String chatWithColab(String message) throws Exception {

        String url = baseUrl + "/api/chat";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, String> body = new HashMap<>();
        body.put("message", message);

        HttpEntity<Map<String, String>> request =
                new HttpEntity<>(body, headers);

        RestTemplate restTemplate = new RestTemplate();

        ResponseEntity<String> response =
                restTemplate.postForEntity(url, request, String.class);

        return response.getBody();
    }

    public String predictNext(java.util.List<String> sequence) throws Exception {

        String url = baseUrl + "/api/predict";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, Object> body = new HashMap<>();
        body.put("sequence", sequence);

        HttpEntity<Map<String, Object>> request =
                new HttpEntity<>(body, headers);

        RestTemplate restTemplate = new RestTemplate();

        ResponseEntity<String> response =
                restTemplate.postForEntity(url, request, String.class);

        return response.getBody();
    }

    public String uploadToPython(
        MultipartFile file
    ) throws Exception {

        String folder = "C:\\Users\\tphuy\\OneDrive\\Documents\\dataset\\";

        File dest = new File(
                folder +
                file.getOriginalFilename()
        );

        file.transferTo(dest);

        String url =
                "http://localhost:5000/process";

        HttpHeaders headers =
                new HttpHeaders();

        headers.setContentType(
                MediaType.APPLICATION_JSON
        );

        Map<String, String> body =
                new HashMap<>();

        body.put(
                "filename",
                file.getOriginalFilename()
        );

        HttpEntity<Map<String, String>>
                request =
                new HttpEntity<>(body, headers);

        RestTemplate restTemplate =
                new RestTemplate();

        ResponseEntity<String> response =
                restTemplate.postForEntity(
                        url,
                        request,
                        String.class
                );

        return response.getBody();
    }
}