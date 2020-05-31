library(keras)
library(tidyr)


data_twitter <- readLines("final/en_US/en_US.twitter.txt", warn = FALSE, encoding = "UTF-8", skipNul = TRUE)

data_twitter <- iconv(data_twitter, "UTF-8", "ASCII", sub="")

data_twitter <- sample(data_twitter, length(data_twitter) * 0.001)


tokenizer <- text_tokenizer() %>% 
  fit_text_tokenizer(data_twitter)

word_index = tokenizer$word_index

vocab_size = length(word_index)

total_words = vocab_size + 1

cat("Found", vocab_size, "unique tokens.\n")


# Create input sequences using list of tokens
input_sequences = list()

for (p in 1:length(data_twitter)){
  token_list = texts_to_sequences(tokenizer, data_twitter[p])[[1]]
  for (J in 1:length(token_list)){
    n_gram_sequence <- replace_na(token_list[1:(J+1)], 0)
    input_sequences <- c(input_sequences, list(n_gram_sequence))
  }
}



# Pad sequences 
max_sequence_len = max(sapply(input_sequences, length))

input_sequences <- as.array(pad_sequences(input_sequences, maxlen = max_sequence_len))



predictors <- input_sequences[, 1:(max_sequence_len-1)]

labels <- input_sequences[, max_sequence_len]
labels <- as.array(labels)

embedding_dim <- 100


k_clear_session()

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = total_words, output_dim = embedding_dim, 
                  input_length = max_sequence_len-1) %>% 
  layer_lstm(128, return_sequences = TRUE) %>%
  layer_dropout(0.2) %>%
  layer_lstm(16) %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = total_words, activation = "softmax")

summary(model)

model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = c("acc")
)



history <- model %>% fit(
  predictors, labels,
  epochs = 120,
  batch_size = 32
  validation_split = 0.1
)


save_model_hdf5(model, "nlp_model.h5")

save_text_tokenizer(tokenizer, "ls")

