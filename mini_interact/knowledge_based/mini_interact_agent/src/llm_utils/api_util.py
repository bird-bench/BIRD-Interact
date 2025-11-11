def read_full_response(stream):
    """
    从 qwen 流式返回中拼接完整的 reasoning_content、content，
    并等到最后带 usage 信息的 chunk 再取 token 用量。
    """
    full_reasoning = ""
    full_content = ""
    token_usage = {}

    for chunk in stream:
        # 如果这个 chunk 带了 usage，就认为它是最后的用量信息，取完就结束
        if (
            getattr(chunk, "usage", None)
            and getattr(chunk.usage, "total_tokens", None) is not None
        ):
            u = chunk.usage
            token_usage = {
                "completion_tokens": u.completion_tokens,
                "prompt_tokens": u.prompt_tokens,
                "total_tokens": u.total_tokens,
                "reasoning_tokens": u.completion_tokens_details.reasoning_tokens,
                "rejected_prediction_tokens": u.completion_tokens_details.rejected_prediction_tokens,
            }
            break

        # 常规 chunk，拼接 delta
        choice = chunk.choices[0]
        delta = choice.delta
        if getattr(delta, "reasoning_content", None):
            full_reasoning += delta.reasoning_content
        if getattr(delta, "content", None):
            full_content += delta.content

    return full_reasoning, full_content, token_usage
