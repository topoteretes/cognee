class ResponseModel {
    @@dynamic
}

// OpenAI client with environment model selection
client<llm> OpenAI {
    provider openai
    options {
        model client_registry.model
        api_key client_registry.api_key
    }
}

function AcreateStructuredOutput(
    text_input: string,
    system_prompt: string,
) -> ResponseModel {
    client OpenAI

    prompt #"
        {{ system_prompt }}
        {{ ctx.output_format }}
        {{ _.role('user') }}
        {{ text_input }}
    "#
}
