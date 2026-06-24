<?php
header('Content-Type: application/json');

$input = json_decode(file_get_contents('php://input'), true);
$userMessage = $input['message'] ?? '';

if (empty($userMessage)) {
    echo json_encode(['error' => 'No message provided']);
    exit;
}

// Environment variables set up via your local Windows dev environment
$geminiKey = getenv('GEMINI_API_KEY');
$cogneeKey = getenv('COGNEE_API_KEY'); 

if (!$geminiKey || !$cogneeKey) {
    die(json_encode(["error" => "Missing system configuration environment keys."]));
}

// Base configuration extracted directly from image_7a6ded.jpg
$tenantUrl = "https://tenant-1e7717fd-de8f-42ba-8a52-7bb1c5d151a8.aws.cognee.ai";
$tenantId  = "1e7717fd-de8f-42ba-8a52-7bb1c5d151a8";

// Standard headers required for Cognee Cloud multi-tenant routing
$cogneeHeaders = [
    "X-Api-Key: " . $cogneeKey,
    "X-Tenant-Id: " . $tenantId,
    "Content-Type: application/json"
];

// ==================================================
// 1. RECALL: Query context insights from your Cognee tenant graph
// ==================================================
$searchPayload = json_encode([
    "query" => $userMessage,
    "search_type" => "GRAPH_COMPLETION",
    "top_k" => 5
]);

$ch = curl_init($tenantUrl . "/api/v1/search");
curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false); // Bypass local Windows/XAMPP curl certificate verification issues
curl_setopt($ch, CURLOPT_HTTPHEADER, $cogneeHeaders);
curl_setopt($ch, CURLOPT_POSTFIELDS, $searchPayload);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$searchResponse = curl_exec($ch);
$searchData = json_decode($searchResponse, true);

$retrievedContext = "No historical context returned.";
if (!empty($searchData)) {
    $retrievedContext = is_array($searchData) ? json_encode($searchData) : $searchData;
}

// ==================================================
// 2. GENERATE: Deliver prompt + memory layer down to Gemini
// ==================================================
$geminiUrl = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key=" . $geminiKey;

$systemPrompt = "You are Rab-bits AI. Here is the historical long-term memory graph context matching the user: " . $retrievedContext;
$fullPrompt = $systemPrompt . "\n\nUser Current Message: " . $userMessage;

$geminiPayload = [
    "contents" => [[
        "parts" => [["text" => $fullPrompt]]
    ]]
];

curl_setopt($ch, CURLOPT_URL, $geminiUrl);
curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($geminiPayload));
$aiResponse = curl_exec($ch);

// ==================================================
// 3. REMEMBER & COGNIFY: Ingest statement asynchronously into your workspace
// ==================================================
$addPayload = json_encode([
    "datasetName" => "rab_bits_dataset",
    "textData" => [$userMessage]
]);

curl_setopt($ch, CURLOPT_URL, $tenantUrl . "/api/v1/add");
curl_setopt($ch, CURLOPT_HTTPHEADER, $cogneeHeaders);
curl_setopt($ch, CURLOPT_POSTFIELDS, $addPayload);
curl_exec($ch);

// Fire process pipeline indexing
curl_setopt($ch, CURLOPT_URL, $tenantUrl . "/api/v1/cognify");
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode(["datasets" => ["rab_bits_dataset"]]));
curl_exec($ch);

curl_close($ch);

// Return final contextualized answer directly to script.js
echo $aiResponse;
?>