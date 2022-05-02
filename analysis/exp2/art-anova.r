library(ARTool)
art_anova = function(formula_string, csv_file_path) {
  myformula <- formula(formula_string)
  print(myformula)
  data <- read.csv(csv_file_path)
  model <- art(myformula, data=data)
  output <- anova(model, response="art")
  
  return(output)
}