const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatArea = document.getElementById('chat-area');
const sendBtn = document.getElementById('send-btn');

function appendMessage(content, sender, isMarkdown = false) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message');
    msgDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
    
    if (isMarkdown) {
        msgDiv.innerHTML = marked.parse(content);
    } else {
        msgDiv.textContent = content;
    }
    
    chatArea.appendChild(msgDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function showTyping() {
    const typingDiv = document.createElement('div');
    typingDiv.classList.add('typing-indicator');
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
    `;
    chatArea.appendChild(typingDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function removeTyping() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;
    
    // Disable input while processing
    userInput.value = '';
    userInput.disabled = true;
    sendBtn.disabled = true;
    sendBtn.style.opacity = '0.5';
    
    // Add user message to UI
    appendMessage(text, 'user');
    
    // Show typing effect
    showTyping();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });
        
        removeTyping();
        
        if (response.ok) {
            const data = await response.json();
            // Parse response as markdown (since Python code added bold/headers)
            appendMessage(data.response, 'bot', true);
        } else {
            appendMessage("Error: Could not connect to the local API. Is the server running?", 'bot');
        }
    } catch (err) {
        removeTyping();
        appendMessage("Network error: Could not reach the server.", 'bot');
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        sendBtn.style.opacity = '1';
        userInput.focus();
    }
});
