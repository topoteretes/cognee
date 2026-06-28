use pyo3::prelude::*;
use std::collections::HashMap;

// Helper to determine if a character is a sentence ending.
fn is_sentence_ending(c: char) -> bool {
    c == '.' || c == ';' || c == '!' || c == '?' || c == '。' || c == '…' || c == '！' || c == '？'
}

// Helper to determine if a character is a paragraph ending.
fn is_paragraph_ending(c: char) -> bool {
    c == '\n' || c == '\r'
}

fn chunk_by_word_rust(data: &str) -> Vec<(String, String)> {
    let mut result = Vec::new();
    let mut current_chunk = String::new();
    let chars: Vec<char> = data.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let character = chars[i];
        current_chunk.push(character);

        if character == ' ' {
            result.push((current_chunk.clone(), "word".to_string()));
            current_chunk.clear();
            i += 1;
            continue;
        }

        if is_sentence_ending(character) {
            let mut next_i = i + 1;
            while next_i < len && chars[next_i] == ' ' {
                current_chunk.push(chars[next_i]);
                next_i += 1;
            }

            let is_paragraph_end = next_i < len && is_paragraph_ending(chars[next_i]);
            let token_type = if is_paragraph_end {
                "paragraph_end"
            } else {
                "sentence_end"
            };
            result.push((current_chunk.clone(), token_type.to_string()));
            current_chunk.clear();
            i = next_i;
            continue;
        }

        i += 1;
    }

    if !current_chunk.is_empty() {
        result.push((current_chunk, "word".to_string()));
    }

    result
}

fn chunk_by_sentence_rust(
    data: &str,
    maximum_size: Option<usize>,
    _py: Python,
    token_counter: &PyAny,
) -> PyResult<Vec<(String, String, usize, Option<String>)>> {
    let words = chunk_by_word_rust(data);
    let mut sentences = Vec::new();
    let mut sentence = String::new();
    let mut paragraph_id = uuid::Uuid::new_v4().to_string();
    let mut sentence_size = 0;
    let mut word_type_state: Option<String> = None;

    // Cache to avoid Python callback overhead for duplicate words
    let mut token_cache: HashMap<String, usize> = HashMap::new();

    for (word, word_type) in words {
        let word_size = if let Some(&size) = token_cache.get(&word) {
            size
        } else {
            let py_res = token_counter.call1((&word,))?;
            let size: usize = py_res.extract()?;
            token_cache.insert(word.clone(), size);
            size
        };

        if word_type == "paragraph_end" || word_type == "sentence_end" {
            word_type_state = Some(word_type.clone());
        } else {
            for character in word.chars() {
                if character.is_alphabetic() {
                    word_type_state = Some(word_type.clone());
                    break;
                }
            }
        }

        if let Some(max_sz) = maximum_size {
            if sentence_size > 0 && sentence_size + word_size > max_sz {
                let cut_type = if word_type_state.as_deref() == Some("word") {
                    "sentence_cut".to_string()
                } else {
                    word_type_state.clone().unwrap_or_else(|| "default".to_string())
                };

                sentences.push((
                    paragraph_id.clone(),
                    sentence.clone(),
                    sentence_size,
                    Some(cut_type),
                ));
                sentence = word;
                sentence_size = word_size;
                continue;
            }
        }

        if word_type == "paragraph_end" || word_type == "sentence_end" {
            sentence.push_str(&word);
            sentence_size += word_size;
            if word_type == "paragraph_end" {
                paragraph_id = uuid::Uuid::new_v4().to_string();
            }

            sentences.push((
                paragraph_id.clone(),
                sentence.clone(),
                sentence_size,
                word_type_state.clone(),
            ));
            sentence.clear();
            sentence_size = 0;
        } else {
            sentence.push_str(&word);
            sentence_size += word_size;
        }
    }

    if !sentence.is_empty() {
        if let Some(max_sz) = maximum_size {
            if sentence_size > max_sz {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Input word longer than chunking size {}",
                    max_sz
                )));
            }
        }
        let section_end = if word_type_state.as_deref() == Some("word") {
            "sentence_cut".to_string()
        } else {
            word_type_state.clone().unwrap_or_else(|| "default".to_string())
        };

        sentences.push((
            paragraph_id,
            sentence,
            sentence_size,
            Some(section_end),
        ));
    }

    Ok(sentences)
}

#[pyfunction]
fn chunk_by_paragraph_rust(
    py: Python,
    data: &str,
    max_chunk_size: usize,
    batch_paragraphs: bool,
    token_counter: &PyAny,
) -> PyResult<Vec<HashMap<String, PyObject>>> {
    let sentences = chunk_by_sentence_rust(data, Some(max_chunk_size), py, token_counter)?;
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut chunk_index = 0;
    let mut paragraph_ids = Vec::new();
    let mut last_cut_type = "default".to_string();
    let mut current_chunk_size = 0;

    let namespace_uuid = uuid::Uuid::parse_str("6ba7b812-9dad-11d1-80b4-00c04fd430c8").unwrap();
    let uuid_module = py.import("uuid")?;

    for (paragraph_id, sentence, sentence_size, end_type) in sentences {
        let end_type_str = end_type.unwrap_or_else(|| "default".to_string());

        if current_chunk_size > 0 && current_chunk_size + sentence_size > max_chunk_size {
            let chunk_uuid_str = uuid::Uuid::new_v5(&namespace_uuid, current_chunk.as_bytes()).to_string();
            let chunk_id_py = uuid_module.call_method1("UUID", (chunk_uuid_str,))?;

            let mut py_paragraph_ids = Vec::new();
            for pid in &paragraph_ids {
                py_paragraph_ids.push(uuid_module.call_method1("UUID", (pid,))?);
            }

            let mut chunk_map = HashMap::new();
            chunk_map.insert("text".to_string(), current_chunk.clone().into_py(py));
            chunk_map.insert("chunk_size".to_string(), current_chunk_size.into_py(py));
            chunk_map.insert("chunk_id".to_string(), chunk_id_py.into_py(py));
            chunk_map.insert("paragraph_ids".to_string(), py_paragraph_ids.into_py(py));
            chunk_map.insert("chunk_index".to_string(), chunk_index.into_py(py));
            chunk_map.insert("cut_type".to_string(), last_cut_type.into_py(py));
            chunks.push(chunk_map);

            paragraph_ids.clear();
            current_chunk.clear();
            current_chunk_size = 0;
            chunk_index += 1;
        }

        paragraph_ids.push(paragraph_id);
        current_chunk.push_str(&sentence);
        current_chunk_size += sentence_size;

        if (end_type_str == "paragraph_end" || end_type_str == "sentence_cut") && !batch_paragraphs {
            let chunk_uuid_str = uuid::Uuid::new_v5(&namespace_uuid, current_chunk.as_bytes()).to_string();
            let chunk_id_py = uuid_module.call_method1("UUID", (chunk_uuid_str,))?;

            let mut py_paragraph_ids = Vec::new();
            for pid in &paragraph_ids {
                py_paragraph_ids.push(uuid_module.call_method1("UUID", (pid,))?);
            }

            let mut chunk_map = HashMap::new();
            chunk_map.insert("text".to_string(), current_chunk.clone().into_py(py));
            chunk_map.insert("chunk_size".to_string(), current_chunk_size.into_py(py));
            chunk_map.insert("chunk_id".to_string(), chunk_id_py.into_py(py));
            chunk_map.insert("paragraph_ids".to_string(), py_paragraph_ids.into_py(py));
            chunk_map.insert("chunk_index".to_string(), chunk_index.into_py(py));
            chunk_map.insert("cut_type".to_string(), end_type_str.clone().into_py(py));
            chunks.push(chunk_map);

            paragraph_ids.clear();
            current_chunk.clear();
            current_chunk_size = 0;
            chunk_index += 1;
        }

        last_cut_type = if end_type_str.is_empty() { "default".to_string() } else { end_type_str };
    }

    if !current_chunk.is_empty() {
        let cut_type = if last_cut_type == "word" { "sentence_cut".to_string() } else { last_cut_type };
        let chunk_uuid_str = uuid::Uuid::new_v5(&namespace_uuid, current_chunk.as_bytes()).to_string();
        let chunk_id_py = uuid_module.call_method1("UUID", (chunk_uuid_str,))?;

        let mut py_paragraph_ids = Vec::new();
        for pid in &paragraph_ids {
            py_paragraph_ids.push(uuid_module.call_method1("UUID", (pid,))?);
        }

        let mut chunk_map = HashMap::new();
        chunk_map.insert("text".to_string(), current_chunk.into_py(py));
        chunk_map.insert("chunk_size".to_string(), current_chunk_size.into_py(py));
        chunk_map.insert("chunk_id".to_string(), chunk_id_py.into_py(py));
        chunk_map.insert("paragraph_ids".to_string(), py_paragraph_ids.into_py(py));
        chunk_map.insert("chunk_index".to_string(), chunk_index.into_py(py));
        chunk_map.insert("cut_type".to_string(), cut_type.into_py(py));
        chunks.push(chunk_map);
    }

    Ok(chunks)
}

#[pymodule]
fn cognee_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chunk_by_paragraph_rust, m)?)?;
    Ok(())
}
