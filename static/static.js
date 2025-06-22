// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // Function to add a message to the chat history
    function addMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `flex ${sender === 'user' ? 'justify-end' : 'justify-start'}`;

        const messageBubble = document.createElement('div');
        // Apply the message-bubble class here for consistency and responsiveness
        messageBubble.className = `p-3 rounded-lg shadow message-bubble ${
            sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'
        }`;

        // Use innerHTML to allow MathJax to process LaTeX and preserve newlines.
        // Replace newlines (\n) with HTML <br> tags for correct rendering of multiline text.
        messageBubble.innerHTML = message.replace(/\n/g, '<br>');

        messageDiv.appendChild(messageBubble);
        chatHistory.appendChild(messageDiv);

        // Trigger MathJax to typeset the new content.
        if (window.MathJax) {
            window.MathJax.typesetPromise([messageDiv]).then(() => {
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }).catch((err) => console.error("MathJax typesetting error:", err));
        } else {
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }

    // Function to handle sending a message
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        addMessage('user', message);
        userInput.value = '';

        // Add a loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'flex justify-start';
        loadingDiv.innerHTML = `
            <div class="bg-gray-200 p-3 rounded-lg shadow-md animate-pulse">
                Thinking...
            </div>
        `;
        chatHistory.appendChild(loadingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.response || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            addMessage('bot', data.response);
        } catch (error) {
            console.error('Error:', error);
            addMessage('bot', `Error: ${error.message}. Please try again.`);
        } finally {
            if (loadingDiv.parentNode) {
                chatHistory.removeChild(loadingDiv);
            }
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});
