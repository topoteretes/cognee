class ResponseModel {
    @@dynamic
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
