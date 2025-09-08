class DynamicInputModel {
    test string
    @@dynamic
}

class DynamicOutputModel {
    test string
    @@dynamic
}

function AcreateStructuredOutput(
    content: string,
    system_prompt: string,
    user_prompt: string,
) -> DynamicOutputModel {
    client OpenAI

    prompt #"
        {{ system_prompt }}
        {{ ctx.output_format }}
        {{ _.role('user') }}
        {{ user_prompt }}
        {{ content }}
    "#
}
