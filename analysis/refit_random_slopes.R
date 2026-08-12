suppressMessages({library(lme4)})
d <- read.csv('main_df.csv')
ctrl <- glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=2e5))
rep <- function(tag, m){
  cat("\n########", tag, "########\n")
  cat("singular:", isSingular(m), " | converged:", is.null(m@optinfo$conv$lme4$messages), "\n")
  if(!is.null(m@optinfo$conv$lme4$messages)) cat("  msg:", m@optinfo$conv$lme4$messages, "\n")
  cat("\n-- variance components --\n"); print(VarCorr(m))
  co <- summary(m)$coefficients
  ci <- confint(m, method='Wald', parm='beta_')
  out <- data.frame(beta=round(co[,1],4), SE=round(co[,2],4),
                    lo=round(ci[,1],4), hi=round(ci[,2],4),
                    z=round(co[,3],2), p=signif(co[,4],3))
  cat("\n-- fixed effects --\n"); print(out)
  invisible(out)
}
t<-Sys.time()
m1 <- glmer(predictShared ~ experiment_num*stance_num*category_num + (1|pid) + (1|matchedDomain) + (1|question),
            data=d, family=binomial, control=ctrl)
cat("M1 fit:", round(as.numeric(difftime(Sys.time(),t,units='secs')),1),"s\n"); rep("M1  + (1|question)", m1)

t<-Sys.time()
m2 <- glmer(predictShared ~ experiment_num*stance_num*category_num + (1+category_num|pid) + (1|matchedDomain) + (1|question),
            data=d, family=binomial, control=ctrl)
cat("\nM2 fit:", round(as.numeric(difftime(Sys.time(),t,units='secs')),1),"s\n"); rep("M2  + (1+category_num|pid)  <-- ROBERT'S ASK", m2)
saveRDS(m2,'m2.rds')
