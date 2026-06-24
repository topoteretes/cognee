const sidebar = document.getElementById('sidebar');
const openSidebar = document.getElementById('openSidebar');
const closeSidebar = document.getElementById('closeSidebar');
const welcome = document.getElementById('welcome_text');
let op = document.querySelector("nav");
let chatContainer = document.querySelector(".chat");
let conversationStarted = false;

openSidebar.addEventListener('click', () => {
    sidebar.classList.add('open');
    openSidebar.style.display = "none";
});

closeSidebar.addEventListener('click', () => {
    sidebar.classList.remove('open');
    openSidebar.style.display = "block";
});

window.addEventListener('click', (event) => {
    if (sidebar.classList.contains('open') && !sidebar.contains(event.target) && !openSidebar.contains(event.target)) {
        sidebar.classList.remove('open');
    }
});

let info = document.getElementById("text");
let searchInput = document.getElementById("search");

/* =========================================
   SEARCH & HISTORY LOGIC
   ========================================= */
function highlightSearch() {
    const searchTerm = searchInput.value.toLowerCase().trim();
    const items = document.querySelectorAll("nav .sidebar-item");

    items.forEach(item => {
        const textSpan = item.querySelector(".sidebar-text");
        const text = textSpan ? textSpan.textContent.toLowerCase() : "";
        const matches = text.includes(searchTerm);

        item.style.display = searchTerm && !matches ? "none" : "flex";

        if (textSpan) {
            textSpan.innerHTML = textSpan.textContent;
            if (searchTerm && matches) {
                const escaped = searchTerm.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                const regex = new RegExp(`(${escaped})`, 'gi');
                textSpan.innerHTML = textSpan.textContent.replace(regex, '<mark>$1</mark>');
            }
        }
    });
}

searchInput.addEventListener('input', highlightSearch);

/* =========================================
   MESSAGE DISPLAY FUNCTIONS
   ========================================= */
function addBotMessage(message) {
    const messageRow = document.createElement("div");

    op.style.display = "none";

    messageRow.className = "chat-message chat-message-bot";
    const messageBubble = document.createElement("div");
    messageBubble.className = "message-bubble";
    messageBubble.style.backgroundColor = "#333";
    messageBubble.textContent = message;
    messageRow.appendChild(messageBubble);
    chatContainer.appendChild(messageRow);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addNavMessage(message) {
    op.style.display = "block";

    const item = document.createElement("div");
    item.className = "sidebar-item";

    const textSpan = document.createElement("span");
    textSpan.className = "sidebar-text";
    textSpan.textContent = message;
    item.appendChild(textSpan);

    const actions = document.createElement("div");
    actions.className = "sidebar-actions";

    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "sidebar-btn edit-btn";
    editBtn.title = "Edit item";
    editBtn.innerHTML = '<i class="fa fa-pencil"></i>';

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "sidebar-btn delete-btn";
    deleteBtn.title = "Delete item";
    deleteBtn.innerHTML = '<i class="fa fa-trash"></i>';

    actions.appendChild(editBtn);
    actions.appendChild(deleteBtn);
    item.appendChild(actions);
    op.appendChild(item);

    deleteBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        item.remove();
    });

    editBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const currentText = textSpan.textContent;
        const input = document.createElement("input");
        input.type = "text";
        input.value = currentText;
        input.className = "sidebar-edit";

        const finishEdit = () => {
            const newValue = input.value.trim();
            if (newValue) {
                textSpan.textContent = newValue;
            }
            item.replaceChild(textSpan, input);
        };

        input.addEventListener("keydown", (keyEvent) => {
            if (keyEvent.key === "Enter") {
                finishEdit();
            }
            if (keyEvent.key === "Escape") {
                item.replaceChild(textSpan, input);
            }
        });

        input.addEventListener("blur", finishEdit);
        item.replaceChild(input, textSpan);
        input.focus();
    });
}

function addChatMessage(message) {
    const messageRow = document.createElement("div");
    messageRow.className = "chat-message chat-message-user";
    const messageBubble = document.createElement("div");
    messageBubble.className = "message-bubble";
    messageBubble.textContent = message;
    messageRow.appendChild(messageBubble);
    chatContainer.appendChild(messageRow);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/* =========================================
   CORE CHAT & INTERCEPTION LOGIC
   ========================================= */
info.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') add();
});

info.addEventListener('focus', () => { welcome.style.display = "none"; });
info.addEventListener('input', () => { welcome.style.display = "none"; });
info.addEventListener('focusout', () => {
    if (!conversationStarted && info.value.trim() === "") welcome.style.display = "block";
});

async function add() {
    const message = info.value.trim();
    if (!message) return;

    conversationStarted = true;
    welcome.style.display = "none";
    const lowerMessage = message.toLowerCase();

    // INTERCEPT IMAGE REQUESTS
    if (lowerMessage.includes("draw") || lowerMessage.includes("image") || lowerMessage.includes("generate")) {
        addNavMessage(message);
        addChatMessage(message);
        generateImage(message); // Fixed Function
        info.value = "";
        return;
    }

    // REGULAR TEXT REQUESTS
    addNavMessage(message);
    addChatMessage(message);
    info.value = "";

    try {
        const response = await fetch('brain.php', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        if (data.candidates) {
            addBotMessage(data.candidates[0].content.parts[0].text);
        }
    } catch (error) {
        console.error("AI Error:", error);
        addBotMessage("Sorry, I'm having trouble connecting right now.");
    }
}

// ===============================
// TOAST HANDLER
// ===============================
const toastTrigger = document.getElementById("liveToastBtn");
const toastLiveExample = document.getElementById("liveToast");

if (toastTrigger) {
    const toastBootstrap = bootstrap.Toast.getOrCreateInstance(toastLiveExample);
    toastTrigger.addEventListener("click", () => {
        toastBootstrap.show();
    });
}

/* =========================================
   GOOGLE SIGN-IN INTEGRATION (FIXED)
   ========================================= */
/**
 * Google Identity Services Initialization & Orchestration
 * Seamlessly manages authentication workflows for local dev and live cloud servers.
 */

window.addEventListener('DOMContentLoaded', () => {
    // 1. Core Verification: Ensure the official Google SDK script is active
    if (typeof google !== 'undefined') {
        google.accounts.id.initialize({
            client_id: "551517105916-ab64kirluov2pll51dve7qmv5la1dqcc.apps.googleusercontent.com",
            callback: handleCredentialResponse
        });

        const googleButton = document.getElementById('google');
        if (googleButton) {
            google.accounts.id.renderButton(googleButton, {
                type: 'standard',
                theme: 'outline',
                size: 'large',
                text: 'signin_with',
                shape: 'rectangular',
                logo_alignment: 'left'
            });

            // Optional prompt for One Tap / auto sign-in if applicable.
            google.accounts.id.prompt();
        } else {
            console.error("Layout Exception: Element with id='google' not found in DOM.");
        }
    } else {
        console.error("Dependency Resolution Exception: Google Identity Services library failed to execute. Ensure script src is present in the document head.");
    }
});

/**
 * Capture the secure Identity Token assertion block and relay it to the database backend.
 * Uses window.location.origin to match XAMPP and Render structures natively.
 * * @param {Object} response - Validated identity asset array from Google API servers.
 */
function handleCredentialResponse(response) {
    const id_token = response.credential;

    if (!id_token) {
        console.warn("Authentication Request Aborted: Empty payload returned by provider identity broker.");
        return;
    }

    // Create a form programmatically to submit the token data securely via POST
    const form = document.createElement('form');
    form.method = 'POST';

    // COMPILER OPTIMIZATION: Dynamically resolve target script matching environment routing
    form.action = `${window.location.origin}/redirect.php`;

    const hiddenField = document.createElement('input');
    hiddenField.type = 'hidden';
    hiddenField.name = 'credential';
    hiddenField.value = id_token;

    form.appendChild(hiddenField);
    document.body.appendChild(form);

    // Execute data payload delivery over to redirect.php
    form.submit();
}


// Grab the DOM elements
const fileInput = document.getElementById('fileInput');
const chat_Container = document.querySelector('.chat');
const welcomeText = document.getElementById('welcome_text');

// Listen for when a file is selected
fileInput.addEventListener('change', function (event) {
    const file = event.target.files[0]; // Get the first uploaded file

    if (file) {
        // 1. Hide the welcome text if it's still there
        if (welcomeText) {
            welcomeText.style.display = 'none';
        }

        // 2. Display the uploaded file in the message section
        addUploadedFileMessage(file.name, file.size);

        // 3. Reset the input value so the same file can be uploaded again if needed
        fileInput.value = '';
    }
});

// Function to create and append the file block into the chat UI
function addUploadedFileMessage(fileName, fileSize) {
    // Convert bytes to a readable format (KB or MB)
    const formattedSize = fileSize > 1024 * 1024
        ? (fileSize / (1024 * 1024)).toFixed(2) + ' MB'
        : (fileSize / 1024).toFixed(2) + ' KB';

    // Create the main message row wrapper (sent by user)
    const messageRow = document.createElement("div");
    messageRow.className = "chat-message chat-message-user";
    messageRow.style.display = "flex";
    messageRow.style.justifyContent = "flex-end";
    messageRow.style.margin = "10px 0";

    // Create the inner file card container
    const fileCard = document.createElement("div");
    fileCard.className = "message-bubble file-bubble";
    fileCard.style.backgroundColor = "#e1f5fe"; // Light blue distinct color for files
    fileCard.style.padding = "10px 15px";
    fileCard.style.borderRadius = "10px";
    fileCard.style.display = "flex";
    fileCard.style.alignItems = "center";
    fileCard.style.maxWidth = "70%";
    fileCard.style.border = "1px solid #b3e5fc";

    // Add a File Icon using your existing FontAwesome setup
    const icon = document.createElement("i");
    icon.className = "fa fa-file";
    icon.style.fontSize = "24px";
    icon.style.marginRight = "12px";
    icon.style.color = "#0288d1";

    // Container for text metadata
    const textContainer = document.createElement("div");
    textContainer.style.display = "flex";
    textContainer.style.flexDirection = "column";

    // File Name Element
    const nameSpan = document.createElement("span");
    nameSpan.className = "file-name";
    nameSpan.textContent = fileName;
    nameSpan.style.fontWeight = "bold";
    nameSpan.style.wordBreak = "break-all";

    // File Size Element
    const sizeSpan = document.createElement("span");
    sizeSpan.className = "file-size";
    sizeSpan.textContent = formattedSize;
    sizeSpan.style.fontSize = "12px";
    sizeSpan.style.color = "#555";

    // Assemble the elements together
    textContainer.appendChild(nameSpan);
    textContainer.appendChild(sizeSpan);

    fileCard.appendChild(icon);
    fileCard.appendChild(textContainer);

    messageRow.appendChild(fileCard);

    // Append the entire row to the chat box
    chat_Container.appendChild(messageRow);

    // Auto-scroll to the bottom of the chat view
    chat_Container.scrollTop = chat_Container.scrollHeight;
}