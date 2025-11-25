def format_step(step: str) -> str:
    """Format a step prompt. Remove leading and trailing whitespaces and newlines, and replace newlines with spaces.
    Also handles cases where LLM generates multiple steps by extracting only the first action/thought.

    Args:
        `step` (`str`): A step prompt in string format.
    Returns:
        `str`: The formatted step prompt.
    """
    step = step.strip('\n').strip()
    
    import re
    
    lines = step.split('\n')
    if len(lines) > 1:
        continuation_patterns = [
            r'^\s*Observation\s*:',
            r'^\s*Thought\s+\d+\s*:',
            r'^\s*Action\s+\d+\s*:',
            r'^\s*Step\s+\d+\s*:'
        ]
        
        result_lines = [lines[0]]
        
        for line in lines[1:]:
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in continuation_patterns):
                break
            if line.strip().startswith('{"type":') and result_lines[0].strip().startswith('{"type":'):
                break
            result_lines.append(line)
        
        step = '\n'.join(result_lines)
    
    return step.replace('\n', ' ').strip()

def format_last_attempt(input: str, scratchpad: str, header: str) -> str:
    return header + f'Input:\n{input}\n' + scratchpad.strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def format_reflections(reflections: list[str], header: str) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_history(history: list[dict]) -> str:
    if history == []:
        return ''
    else:
        return '\n' + '\n'.join([f"Command: {turn['command']}\nObservation: {turn['observation']}\n" for turn in history]) + '\n'

def format_chat_history(history: list[tuple[str, str]]) -> str:
    if history == []:
        return 'No chat history.\n'
    else:
        return '\n' + '\n'.join([f"{role.capitalize()}: {chat}" for chat, role in history]) + '\n'

def str2list(s: str) -> list[int]:
    return [int(i) for i in s.split(',')]

def get_avatar(agent_type: str) -> str:
    if 'reflector' in agent_type.lower():
        return '👩‍🔬'
    elif 'analyst' in agent_type.lower():
        return '👩‍💻'
    else:
        return '🤖'
