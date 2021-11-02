x为最优值点
out.fval为最优函数值
out.fvec存储迭代过程中函数值

	* |opts.maxit| ：最大外层迭代次数
	* |opts.maxit\_inn| ：最大内层迭代次数
	* |opts.ftol| ：针对函数值的停机判断条件
	* |opts.gtol| ：针对梯度的停机判断条件
	* |opts.factor| ：正则化系数的衰减率
	* |opt.mu1| ：初始的正则化系数
	* |opts.alpha0| ：初始步长
	* |opts.opts1| ：结构体，用于向内层算法提供其它具体的参数
	* |opts.thres| ：判断小量是否被认为为 $0$ 的阈值
	* |opts.bb| ：是否使用BB步长
	* |opts1.maxit| ：最大(内部)迭代次数
	* |opts.alpha0| ：步长的初始值
	* |optsz.verbose| ：不为 0 时输出每步迭代信息，否则不输出
	* |opts.ls| ：标记是否线搜索
	* |opts.sigma| ：Huber 光滑化参数 $\sigma$