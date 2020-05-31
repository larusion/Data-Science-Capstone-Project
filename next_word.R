library(keras)

model <- load_model_hdf5("nlp_model.h5")

tokenizer <- load_text_tokenizer("ls")

word_index = tokenizer$word_index

max_sequence_len <- 34

seed_text = "I need"
seed_text <- iconv(seed_text, "UTF-8", "ASCII", sub="")

token <- texts_to_sequences(tokenizer, seed_text)[1]

token <- as.array(pad_sequences(token, maxlen = max_sequence_len-1))

predicted = model %>% predict_classes(token, verbose = 0)

output_word <- names(word_index[predicted])

seed_text = paste(seed_text, output_word)

print(seed_text)

