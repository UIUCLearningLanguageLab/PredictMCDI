# Title     : get_CHILDES
# Objective : Download CHILDES utterances including adult and child directed speech.
# Created by: andrewzflores
# Created on: 10/7/19

install.packages("childesr")
library(childesr)

#Download CHILDES utterances
childes_utterance <- get_utterances(collection = "Eng-NA")

#Include only relevant age range (0-30 months)
childes_utterance <- childes_utterance %>% filter (target_child_age >= 0 & target_child_age <= 31)

#Include only relevant columns and save output.
childes_utterance <- childes_utterance %>% select(id,transcript_id,gloss,stem, speaker_code, target_child_age, type)
write.csv(childes_utterance, file = "childes_utterances.csv", row.names = F)
