function addBotMessage(message) 
{
    const messageRow = document.createElement("div");
    messageRow.className = "chat-message chat-message-bot"; 
    const messageBubble = document.createElement("div");
    messageBubble.className = "message-bubble";
    messageBubble.textContent = message;
    messageRow.appendChild(messageBubble);
    chatContainer.appendChild(messageRow);
}
const { GoogleGenerativeAI } = require("@google/generative-ai");

// This pulls the key you just saved in the terminal
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

async function chat() 
{
  const prompt = "Hello Gemini! I'm building a chatboard. Are you connected?";
  
  try 
  {
    const result = await model.generateContent(prompt);
    console.log("Gemini says: " + result.response.text());
  } 
  catch (error) 
  {
    console.error("Connection failed. Check if your API key is active.");
  }
}

chat();


const messageRow = document.createElement("div");
messageRow.className = "chat-message chat-message-bot";