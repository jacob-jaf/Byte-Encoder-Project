use std::collections::HashMap;
use onig::Regex;
use itertools::Itertools;

pub fn encode(
    re: &Regex, // Regex to pre-tokenize
    vocab_inv_bytes: &[Option<u16>],  // Mapping from byte to token for initial vocab
    merges: &HashMap<(u16, u16), u16>, // Tuple of tokens to merged token
    text: &str
) -> Vec<u16> {
    let words: Vec<&str> = re.find_iter(text).map(|m| &text[m.0..m.1]).collect();
    words.into_iter().flat_map(|word| {
        let word_bytes = word.as_bytes();
        let mut symbols: Vec<u16> = word_bytes.into_iter().map(|c| vocab_inv_bytes[*c as usize].unwrap()).collect();

        loop {
            let candidate_merges = symbols.iter().tuple_windows().enumerate().filter_map(|(i, (a, b))| {
                merges.get(&(*a, *b)).map(|v| (i, *v))
            });
            let best_merge = candidate_merges.min_by_key(|(_index, merged_token)| *merged_token); // Earliest merge in list of merges

            if let Some((merge_index, merge_token)) = best_merge {
                symbols[merge_index] = merge_token;
                symbols.remove(merge_index + 1);  // O(n) worst case
            } else {
                break;
            }
        }
        // println!("Merged {:?} into {:?}", word, word_bytes, symbols)

        symbols
    }).collect()
}


pub fn decode(v: &[u16], vocab: &HashMap<u16, Vec<u8>>) -> Vec<u8> {
    v.into_iter().flat_map(|&token| vocab.get(&token).unwrap()).copied().collect()
}
