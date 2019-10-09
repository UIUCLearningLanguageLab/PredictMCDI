# Title     : clean_mcdi
# Objective : 1)To remove/replace items that meet certain criteria. Dichotomize "value" for future analysis.
#             2)To calculate MCDIp (proportion of children who produce a word for each word at each age.) 
# Created by: andrewzflores
# Created on: 10/3/19

install.packages("tidyverse")
library(tidyverse)



mcdi_data <- read.csv("mcdi.csv")

# Remove undesired item types. 
exclude_types <- c("how_use_words","complexity","word_endings",
                   "word_forms_nouns","word_forms_verbs","combine",
                   "word_endings_nouns","word_endings_verbs")

mcdi_data <- mcdi_data %>% filter(!type %in% exclude_types)

#remove definitions based on the following criteria:
#1)Remove from data if definition is compound word (e.g play dough).
#2)Include only a single item for words with multiple meaning (e.g can(object, can (auxiliary))) 
remove_items <- c("baa baa", "choo choo", "quack quack", "uh oh", "woof woof", "yum yum",
                  "play dough", "chicken (food)", "fish (food)", "french fries", "green beans", 
                  "ice cream", "peanut butter", "potato chip", "belly button", "TV", "high chair",
                  "living room", "rocking chair", "washing machine", "play pen", "lawn mower",
                  "gas station", "babysitter's name", "child's own name", "pet's name", "I", 
                  "give me five!","gonna get you!","go potty","night night",
                  "so big!","thank you","this little piggy","turn around","drink (action)",
                  "slide (action)", "swing (action)","watch (action)","work (action)","all gone",
                  "clean (description)","dry (description)","orange (description)","next to",
                  "on top of","a lot","can (auxiliary)","don't","water (not beverage)")

mcdi_data <- mcdi_data %>% filter(!definition %in% remove_items)


#Replaces specific word meaning (e.g "drink (beverage)" with lemma (drink))
mcdi_data$definition <- recode(mcdi_data$definition, "chicken (animal)" = "chicken", 
                               "fish (animal)" = "fish", "toy (object)" = "toy", 
                               "drink (beverage)" = "drink","orange (food)" = "orange",
                               "water (beverage)" = "water", "dress (object)" = "dress",
                               "buttocks/bottom*" = "bottom", "owie/boo boo" = "owie", 
                               "penis*" = "penis", "vagina*" = "vagina", "can (object)" = "can",
                               "tissue/kleenex" = "tissue", "watch (object)" = "watch",
                               "slide (object)" = "slide", "swing (object)" = "swing" , 
                               "church*" = "church", "work (place)" = "work", "daddy*" = "daddy",
                               "grandma*" = "grandma", "grandpa*" = "granpa", "mommy*" = "mommy",
                               "call (on phone)" = "call", "clean (action)" = "clean", 
                               "dry (action)" = "dry", "little (description)" = "little",
                               "inside/in"= "inside", "did/did ya" = "did", "gonna/going to" = "gonna",
                               "gotta/got to" = "gotta", "hafta/have to" = "hafta", "lemme/let me" = "lemme",
                               "need/need to" = "need", "try/try to" = "try", "wanna/want to" = "wanna",
                               "shh/shush/hush" = "hush", "soda/pop" = "soda",  )
                    
#Dichotomize "value". (produces = 1, else = 0)
mcdi_data$value <- as.character(mcdi_data$value)
mcdi_data$value[mcdi_data$value == "produces"] <- 1
mcdi_data$value[mcdi_data$value == ""] <- 0

#Calculate mcdip.
mcdi_data$value <- as.numeric(mcdi_data$value)
MCDIp <- mcdi_data %>% group_by(age,definition) %>% summarize(MCDIp = mean(value,na.rm=TRUE))

#Merge with clean_mcdi data.
mcdi_data <- left_join(mcdi_data, MCDIp)

#Include only relevant columns and save cleaned mcdi.

mcdi_data <- mcdi_data %>% select(data_id, age, definition, value,MCDIp,lexical_class)
write.csv(mcdi_data, file = "clean_mcdi.csv", row.names = F)

#Save a list of the unique words in our data.

mcdi_target <- unique(mcdi_data$definition)
mcdi_target <- as.data.frame(mcdi_target)
colnames(mcdi_target) <- c("words")
write.table(mcdi_target, file = "target_mcdi.csv", row.names = F, col.names= TRUE)




             

 
             