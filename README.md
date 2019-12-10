# ePcl-
Web Usage Mining Project, studying users' interactions with a website. 

Three main functions implemented : 

1) Predicting web user's next move : model_Q_Learning 
  Original Paper : Taghipour, N., Kardan, A., Ghidary, S.S.: Usage-based web recommendations: a reinforcement learning approach. In: Proceedings of the 2007 ACM Conference on Recommender Systems, RecSys 2007, Minneapolis, MN, USA, October 19-20, 2007, pp. 113–120 (2007). 
	
2) Finding communities based on web paths' shapes 
	Original Papers : 
		S. Jouili and S. Tabbone. Graph matching using node signatures. In IAPR-TC15 Workshop on GbRPR, Italy, LNCS 5534, Springer, pages 154–163, 2009.
		D. P. Lopresti and G. T. Wilfong. A fast technique for comparing graph representations with applications to performance evaluation. IJDAR, 6(4) : 219–229, 2003. 
	Carries out a graph embedding using two different graph representation methods, and then applies an agglomerative clustering. 
	
3) Infering subscription probability based on past sessions : 
	Builds a Bayesian network using the Chow-Liu algorithm on the explanatory variables and infers on the subscription variable using a maximum a posteriori estimation. 
	
