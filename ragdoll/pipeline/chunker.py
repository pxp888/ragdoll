import re
from typing import List, Dict, Any
from ragdoll.clients.kev import KevClient


def split_coarse_chunks(
    text: str,
    min_chars: int = 2000,
    max_chars: int = 4000,
) -> List[Dict[str, Any]]:
    """
    Stage 1: Coarse Split (2k-4k chars, structural anchors).
    Partitions text at structural boundaries (headers, double newlines, etc.).
    """
    if not text.strip():
        return []

    # Encode to bytes for byte offsets calculation
    text_bytes = text.encode("utf-8")

    # Split into structural blocks (double newlines or headers)
    # Using regex splitting with capturing delimiter to keep formatting
    pattern = r"(\n\s*\n+|(?<=\n)(?=[#]{1,6}\s))"
    raw_pieces = re.split(pattern, text)

    blocks: List[str] = []
    # Recombine pieces if split created empty or isolated newlines
    temp = ""
    for piece in raw_pieces:
        if not piece:
            continue
        temp += piece
        if len(temp) >= 200:  # avoid micro-blocks
            blocks.append(temp)
            temp = ""
    if temp:
        if blocks:
            blocks[-1] += temp
        else:
            blocks.append(temp)

    # Now group blocks into chunks between min_chars and max_chars
    chunks: List[Dict[str, Any]] = []
    current_text = ""
    current_char_start = 0

    def add_chunk(chunk_str: str, char_start: int):
        char_end = char_start + len(chunk_str)
        byte_start = len(text[:char_start].encode("utf-8"))
        byte_end = len(text[:char_end].encode("utf-8"))
        chunks.append({
            "chunk_index": len(chunks),
            "text": chunk_str,
            "byte_offset_start": byte_start,
            "byte_offset_end": byte_end,
        })

    for block in blocks:
        # If adding block exceeds max_chars and current_text is >= min_chars
        if current_text and (len(current_text) + len(block) > max_chars):
            if len(current_text) >= min_chars:
                add_chunk(current_text, current_char_start)
                current_char_start += len(current_text)
                current_text = block
                continue

        # If single block is itself longer than max_chars, split on lines/sentences
        if len(block) > max_chars:
            if current_text:
                add_chunk(current_text, current_char_start)
                current_char_start += len(current_text)
                current_text = ""

            sub_units = re.split(r"(\n+|(?<=[.?!])\s+)", block)
            sub_accum = ""
            for su in sub_units:
                if len(sub_accum) + len(su) > max_chars and len(sub_accum) >= min_chars:
                    add_chunk(sub_accum, current_char_start)
                    current_char_start += len(sub_accum)
                    sub_accum = su
                else:
                    sub_accum += su
            if sub_accum:
                current_text = sub_accum
        else:
            current_text += block

    if current_text.strip():
        add_chunk(current_text, current_char_start)

    return chunks


def split_atomic_units(text: str) -> List[str]:
    """
    Splits text into small sentence or line units (~50-150 chars).
    Preserves all characters so joining equals original text.
    """
    pattern = r"((?<=[.?!])\s+|\n+)"
    parts = re.split(pattern, text)
    units = []
    curr = ""
    for p in parts:
        curr += p
        # Form units of around 80-120 chars if possible
        if len(curr) >= 80:
            units.append(curr)
            curr = ""
    if curr:
        if units:
            units[-1] += curr
        else:
            units.append(curr)
    return units


async def split_fine_chunks_with_kev(
    coarse_chunk: Dict[str, Any],
    kev_client: KevClient,
    min_chars: int = 300,
    max_chars: int = 600,
) -> List[Dict[str, Any]]:
    """
    Worker function: Scans a single coarse chunk, dividing it
    into fine chunks (~300-600 characters) via Kev multi-candidate breakpoint decisions.
    Retains parent_chunk_id = coarse_chunk['id'].
    """
    coarse_text = coarse_chunk["text"]
    coarse_byte_start = coarse_chunk["byte_offset_start"]
    parent_id = coarse_chunk.get("id")

    units = split_atomic_units(coarse_text)
    if not units:
        return []

    fine_chunks: List[Dict[str, Any]] = []
    current_block = ""
    current_char_start = 0

    def add_fine_chunk(chunk_str: str, char_start: int):
        char_end = char_start + len(chunk_str)
        rel_byte_start = len(coarse_text[:char_start].encode("utf-8"))
        rel_byte_end = len(coarse_text[:char_end].encode("utf-8"))
        fine_chunks.append({
            "parent_chunk_id": parent_id,
            "chunk_index": len(fine_chunks),
            "text": chunk_str,
            "byte_offset_start": coarse_byte_start + rel_byte_start,
            "byte_offset_end": coarse_byte_start + rel_byte_end,
        })

    i = 0
    n = len(units)
    while i < n:
        unit = units[i]
        if not current_block:
            current_block = unit
            i += 1
            continue

        cand_len = len(current_block) + len(unit)

        # Case 1: Below minimum threshold, accumulate without calling Kev
        if len(current_block) < min_chars:
            current_block += unit
            i += 1
            continue

        # Case 2: Above maximum threshold, force split
        if cand_len > max_chars:
            add_fine_chunk(current_block, current_char_start)
            current_char_start += len(current_block)
            current_block = unit
            i += 1
            continue

        # Case 3: In [min_chars, max_chars], collect candidate units up to max_chars
        candidates: List[str] = []
        accum_len = len(current_block)
        cand_idx = i
        while cand_idx < n:
            u = units[cand_idx]
            if accum_len + len(u) <= max_chars:
                candidates.append(u)
                accum_len += len(u)
                cand_idx += 1
            else:
                break

        if not candidates:
            add_fine_chunk(current_block, current_char_start)
            current_char_start += len(current_block)
            current_block = unit
            i += 1
            continue

        # Query Kev for breakpoint decision among candidates
        if hasattr(kev_client, "decide_breakpoint"):
            res = await kev_client.decide_breakpoint(
                preceding_text=current_block.strip(),
                candidates=[c.strip() for c in candidates],
            )
            split_count = res if isinstance(res, int) else 0
        else:
            is_split = await kev_client.decide_boundary(
                block_a=current_block.strip(),
                block_b=candidates[0].strip(),
            )
            split_count = 0 if is_split else len(candidates)

        if split_count == 0:
            # Split before candidates (current_block forms complete chunk)
            add_fine_chunk(current_block, current_char_start)
            current_char_start += len(current_block)
            current_block = ""
        elif split_count < len(candidates):
            # Split after candidate k
            chosen_units = "".join(candidates[:split_count])
            full_chunk = current_block + chosen_units
            add_fine_chunk(full_chunk, current_char_start)
            current_char_start += len(full_chunk)
            current_block = ""
            i += split_count
        else:
            # Continue all candidates without splitting
            current_block += "".join(candidates)
            i += len(candidates)

    if current_block.strip():
        # If trailing block is too small and there's a previous chunk, merge it
        if fine_chunks and len(current_block) < (min_chars // 2):
            prev = fine_chunks[-1]
            prev["text"] += current_block
            rel_byte_end = len(coarse_text[:current_char_start + len(current_block)].encode("utf-8"))
            prev["byte_offset_end"] = coarse_byte_start + rel_byte_end
        else:
            add_fine_chunk(current_block, current_char_start)

    return fine_chunks
