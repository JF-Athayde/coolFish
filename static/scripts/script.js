async function enviarMensagem() {
    const input = document.getElementById("entrada");
    const texto = input.value.trim();
    if(!texto) return;

    const chat = document.getElementById("chat");

    // Mensagem do usuário
    const msgUsuario = document.createElement("div");
    msgUsuario.className = "mensagem usuario";
    msgUsuario.textContent = texto;
    chat.appendChild(msgUsuario);

    input.value = "";

    // Enviar para backend
    const resposta = await fetch("/mensagem", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mensagem: texto })
    }).then(res => res.json());

    // Mensagem do assistente
    const msgAssistente = document.createElement("div");
    msgAssistente.className = "mensagem assistente";
    msgAssistente.textContent = resposta.resposta;
    chat.appendChild(msgAssistente);

    // Scroll automático
    chat.scrollTop = chat.scrollHeight;
}

document.getElementById("entrada").addEventListener("keydown", function(e) {
    if(e.key === "Enter") enviarMensagem();
});