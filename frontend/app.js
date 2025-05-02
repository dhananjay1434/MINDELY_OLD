// Configuration
const API_URL = 'https://your-backend-url.herokuapp.com/api'; // Replace with your actual backend URL

// DOM Elements
const authContainer = document.getElementById('auth-container');
const chatContainer = document.getElementById('chat-container');
const loginTab = document.getElementById('login-tab');
const registerTab = document.getElementById('register-tab');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const loginBtn = document.getElementById('login-btn');
const registerBtn = document.getElementById('register-btn');
const loginError = document.getElementById('login-error');
const registerError = document.getElementById('register-error');
const logoutBtn = document.getElementById('logout-btn');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');

// State
let currentUser = null;

// Check if user is already logged in
function checkAuth() {
    const userId = localStorage.getItem('userId');
    if (userId) {
        currentUser = { userId };
        showChat();
        loadChatHistory();
    }
}

// Switch between login and register tabs
loginTab.addEventListener('click', () => {
    loginTab.classList.add('active');
    registerTab.classList.remove('active');
    loginForm.classList.remove('hidden');
    registerForm.classList.add('hidden');
});

registerTab.addEventListener('click', () => {
    registerTab.classList.add('active');
    loginTab.classList.remove('active');
    registerForm.classList.remove('hidden');
    loginForm.classList.add('hidden');
});

// Handle login
loginBtn.addEventListener('click', async () => {
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;
    
    if (!username || !password) {
        loginError.textContent = 'Please enter both username and password';
        return;
    }
    
    loginBtn.disabled = true;
    loginBtn.textContent = 'Logging in...';
    
    try {
        const response = await fetch(`${API_URL}/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentUser = { userId: data.user_id };
            localStorage.setItem('userId', data.user_id);
            showChat();
            loadChatHistory();
        } else {
            loginError.textContent = data.message || 'Login failed';
        }
    } catch (error) {
        loginError.textContent = 'Connection error. Please try again.';
        console.error('Login error:', error);
    } finally {
        loginBtn.disabled = false;
        loginBtn.textContent = 'Login';
    }
});

// Handle registration
registerBtn.addEventListener('click', async () => {
    const username = document.getElementById('register-username').value.trim();
    const password = document.getElementById('register-password').value;
    const confirmPassword = document.getElementById('register-confirm').value;
    
    if (!username || !password) {
        registerError.textContent = 'Please enter both username and password';
        return;
    }
    
    if (password !== confirmPassword) {
        registerError.textContent = 'Passwords do not match';
        return;
    }
    
    registerBtn.disabled = true;
    registerBtn.textContent = 'Registering...';
    
    try {
        const response = await fetch(`${API_URL}/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Switch to login tab
            loginTab.click();
            document.getElementById('login-username').value = username;
            loginError.textContent = 'Registration successful! Please log in.';
            loginError.style.color = '#2ecc71';
        } else {
            registerError.textContent = data.message || 'Registration failed';
        }
    } catch (error) {
        registerError.textContent = 'Connection error. Please try again.';
        console.error('Registration error:', error);
    } finally {
        registerBtn.disabled = false;
        registerBtn.textContent = 'Register';
    }
});

// Handle logout
logoutBtn.addEventListener('click', () => {
    localStorage.removeItem('userId');
    currentUser = null;
    showAuth();
    clearChat();
});

// Handle sending messages
sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Clear input
    messageInput.value = '';
    
    // Add user message to chat
    addMessage('user', message);
    
    // Show typing indicator
    const typingIndicator = addTypingIndicator();
    
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: currentUser.userId,
                message
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        typingIndicator.remove();
        
        if (data.success) {
            addMessage('assistant', data.response);
        } else {
            addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
            console.error('API error:', data.message);
        }
    } catch (error) {
        // Remove typing indicator
        typingIndicator.remove();
        
        addMessage('assistant', 'Sorry, I encountered a connection error. Please check your internet connection and try again.');
        console.error('Send message error:', error);
    }
}

// Load chat history
async function loadChatHistory() {
    try {
        const response = await fetch(`${API_URL}/history?user_id=${currentUser.userId}`);
        const data = await response.json();
        
        if (data.success && data.history.length > 0) {
            clearChat();
            
            // Add welcome message
            addMessage('assistant', 'Hi there! I\'m Mandy, CEO of Mindely. How are you doing today?');
            
            // Add history messages
            data.history.forEach(msg => {
                addMessage(msg.role, msg.content);
            });
        }
    } catch (error) {
        console.error('Load history error:', error);
    }
}

// UI Helper Functions
function showAuth() {
    authContainer.classList.remove('hidden');
    chatContainer.classList.add('hidden');
}

function showChat() {
    authContainer.classList.add('hidden');
    chatContainer.classList.remove('hidden');
}

function clearChat() {
    chatMessages.innerHTML = '';
    addMessage('assistant', 'Hi there! I\'m Mandy, CEO of Mindely. How are you doing today?');
}

function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', role);
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    
    const paragraph = document.createElement('p');
    paragraph.textContent = content;
    
    contentDiv.appendChild(paragraph);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'assistant', 'typing-indicator');
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    
    const paragraph = document.createElement('p');
    paragraph.textContent = 'Typing...';
    
    contentDiv.appendChild(paragraph);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

// Initialize
checkAuth();
