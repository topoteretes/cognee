<?php
session_start();

// 1. Grab the token sent by Google via the POST method
$id_token = $_POST['credential'] ?? null;

if (!$id_token) {
    die("Authentication failed: No token received.");
}

// 2. Break the token into parts (Header, Payload, Signature)
$token_parts = explode('.', $id_token);

if (count($token_parts) === 3) {
    // 3. Decode the middle part (the Payload) which contains user details
    $payload_encoded = $token_parts[1];
    
    // Fix base64 padding issues that sometimes happen in PHP
    $payload_decoded = base64_decode(str_replace(['-', '_'], ['+', '/'], $payload_encoded));
    
    // 4. Convert the JSON string into a usable PHP Array
    $user_data = json_decode($payload_decoded, true);

    // 5. Extract user details safely
    $email = $user_data['email'];
    $name = $user_data['name'];
    $google_id = $user_data['sub']; // Unique ID from Google
    $picture = $user_data['picture'];

    // 6. Log the user into your system session
    $_SESSION['user_logged_in'] = true;
    $_SESSION['user_email'] = $email;
    $_SESSION['user_name'] = $name;
    $_SESSION['user_picture'] = $picture;

    // 7. Redirect them to your main dashboard or homepage
    header("Location: index.html");
    exit();

} else {
    die("Invalid token structure received.");
}
?>