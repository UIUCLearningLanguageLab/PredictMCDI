# Title     : get_mcdi
# Objective : Download and save mcdi file.
# Created by: andrewzflores
# Created on: 10/3/19

install.packages("wordbankr")
library(wordbankr)

# Download mcdi "Words & Sentences"
get_mcdi <- get_instrument_data("English (American)", "WS", administrations = TRUE, iteminfo = TRUE)

#Save file.
write.csv(get_mcdi, file= "mcdi.csv", row.names = FALSE)
