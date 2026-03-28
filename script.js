import { pipeline, cos_sim } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatArea = document.getElementById('chat-area');
const sendBtn = document.getElementById('send-btn');

let extractor = null;
let kbIndexed = [];
let isReady = false;

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

async function initializeAI() {
    showTyping();
    appendMessage("Initializing lightweight AI model right in your browser! This executes totally offline and happens only once... 🚀", "bot");
    
    try {
        // Download and load model locally in browser WebAssembly
        extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
            revision: 'default'
        });
        
        // Fetch Knowledge Base
        const response = await fetch('./github_kb.json');
        const kb = await response.json();
        
        // Pre-compute Embeddings
        for (const entry of kb) {
            const searchText = `${entry.feature} ${entry.description} ${entry.keywords.join(' ')}`;
            const output = await extractor(searchText, { pooling: 'mean', normalize: true });
            kbIndexed.push({
                ...entry,
                embedding: Array.from(output.data) // Convert Float32Array
            });
        }
        
        removeTyping();
        appendMessage("✨ AI Model initialized! Knowledge base vector indexing complete. How can I help you?", "bot");
        
        isReady = true;
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
        
    } catch (err) {
        removeTyping();
        appendMessage("❌ Error: Failed to load the AI model or knowledge base. Check browser console.", "bot");
        console.error(err);
    }
}

// Generate the final markdown response based on the dataset entry
function generateResponse(matchedEntry) {
    if (!matchedEntry) {
        return "Sorry, I don't have information on that in my structured dataset. Please check GitHub Docs: [https://docs.github.com/](https://docs.github.com/)";
    }
    
    let response = `### ${matchedEntry.feature}\n\nHere are the instructions to help you:\n`;
    matchedEntry.steps.forEach((step, index) => {
        response += `${index + 1}. ${step}\n`;
    });
    response += `\n**Official Documentation**: [Link](${matchedEntry.url})`;
    return response;
}

// Intercept Chat Form Submit
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!isReady) return;
    
    const text = userInput.value.trim();
    if (!text) return;
    
    userInput.value = '';
    userInput.disabled = true;
    sendBtn.disabled = true;
    
    appendMessage(text, 'user');
    showTyping();
    
    try {
        // Embed user question
        const questionOutput = await extractor(text, { pooling: 'mean', normalize: true });
        const questionEmbedding = Array.from(questionOutput.data);
        
        // Find highest cosine similarity in the knowledge base
        let bestScore = -1;
        let bestMatch = null;
        
        for (const entry of kbIndexed) {
            const score = cos_sim(questionEmbedding, entry.embedding);
            if (score > bestScore) {
                bestScore = score;
                bestMatch = entry;
            }
        }
        
        removeTyping();
        
        // If match threshold > 0.45 
        if (bestScore >= 0.45) {
            const responseText = generateResponse(bestMatch);
            appendMessage(responseText, 'bot', true);
        } else {
            appendMessage(generateResponse(null), 'bot', true);
        }
        
    } catch (err) {
        removeTyping();
        appendMessage("An issue occurred computing the local embeddings. 😢", 'bot');
        console.error(err);
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
});

// Run AI Initialization on load
window.addEventListener('load', () => {
    initializeAI();
});
