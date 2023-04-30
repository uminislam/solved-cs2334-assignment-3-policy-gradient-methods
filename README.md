Download Link: https://assignmentchef.com/product/solved-cs2334-assignment-3-policy-gradient-methods
<br>
<h1>1           Policy Gradient Methods</h1>

The goal of this problem is to experiment with policy gradient and its variants, including variance reduction methods. Your goals will be to set up policy gradient for both continuous and discrete environments, and implement a neural network baseline for variance reduction. The framework for the vanilla policy gradient algorithm is setup in the starter code pg.py, and everything that you need to implement is in this file. The file has detailed instructions for each implementation task, but an overview of key steps in the algorithm is provided here. For this assignment you need to have <a href="http://www.mujoco.org/index.html">MuJoCo</a> installed, please follow the <a href="https://drive.google.com/file/d/1PriAh0D3QSp2-5jLed-UohX9qH8j51zj/view">installation guide</a><a href="https://drive.google.com/file/d/1PriAh0D3QSp2-5jLed-UohX9qH8j51zj/view">.</a>

<h2>REINFORCE</h2>

Recall the vanilla policy-gradient theorem,

∇<em><sub>θ</sub>J</em>(<em>θ</em>) = E<em>π<sub>θ </sub></em>[∇<em><sub>θ </sub></em>log<em>π<sub>θ</sub></em>(<em>a</em>|<em>s</em>)<em>Q<sup>π</sup></em><em><sup>θ</sup></em>(<em>s,a</em>)]

REINFORCE is a monte-carlo policy gradient algorithm, so we will be using the sampled returns <em>G<sub>t </sub></em>as unbiased estimates of <em>Q<sup>π</sup></em><em><sup>θ</sup></em>(<em>s,a</em>). Then the gradient update can be expressed as maximizing the following objective function: where <em>D </em>is the set of all trajectories collected by policy <em>π<sub>θ</sub></em>, and <em>τ </em>= (<em>s</em><sub>0</sub><em>,a</em><sub>0</sub><em>,r</em><sub>0</sub><em>,s</em><sub>1</sub><em>…,s<sub>T</sub></em>) is a trajectory.

<h2>Baseline</h2>

One difficulty of training with the REINFORCE algorithm is that the monte-carlo estimated return <em>G<sub>t </sub></em>can have high variance. To reduce variance, we subtract a baseline <em>b<sub>φ</sub></em>(<em>s</em>) from the estimated returns when computing the policy gradient. A good baseline is the state value function parametrized by <em>φ</em>, <em>b<sub>φ</sub></em>(<em>s</em>) = <em>V <sup>π</sup></em><em><sup>θ</sup></em>(<em>s</em>), which requires a training update to <em>φ </em>to minimize the following mean-squared error loss:

1

<h2>Advantage Normalization</h2>

After subtracting the baseline, we get the following new objective function:

where

<em>A</em>ˆ<em>t </em>= <em>G</em><em>t </em>− <em>b</em><em>φ</em>(<em>s</em><em>t</em>)

A second variance reduction technique is to normalize the computed advantages, <em>A</em>ˆ<em><sub>t</sub></em>, so that they have mean 0 and standard deviation 1. From a theoretical perspective, we can consider centering the advantages to be simply adjusting the advantages by a constant baseline, which does not change the policy gradient. Likewise, rescaling the advantages effectively changes the learning rate by a factor of 1<em>/σ</em>, where <em>σ </em>is the standard deviation of the empirical advantages.

<h2>1.1         Coding Questions</h2>

The functions that you need to implement in pg.py are enumerated here. Detailed instructions for each function can be found in the comments in pg.py. We strongly encourage you to look at pg.py and understand the code structure first.

<ul>

 <li>buildmlp</li>

 <li>addplaceholdersop</li>

 <li>buildpolicynetworkop</li>

 <li>addlossop</li>

 <li>addoptimizerop</li>

 <li>addbaselineop</li>

 <li>getreturns</li>

 <li>calculateadvantage</li>

 <li>updatebaseline</li>

</ul>

<h2>1.2         Writeup Questions</h2>

<ul>

 <li>(4 pts) (CartPole-v0) Test your implementation on the CartPole-v0 environment by running</li>

</ul>

With the given configuration file config.py, the average reward should reach 200 within 100 iterations. <em>NOTE: training may repeatedly converge to 200 and diverge. Your plot does not have to reach 200 and stay there. We only require that you achieve a perfect score of 200 sometime during training.</em>

Include in your writeup the tensorboard plot for the average reward. Start tensorboard with:

and then navigate to the link it gives you. Click on the “SCALARS” tab to view the average reward graph.

Now, test your implementation on the CartPole-v0 environment without baseline by running

Include the tensorboard plot for the average reward. Do you notice any difference? Explain.

<ul>

 <li>(4 pts) (InvertedPendulum-v1) Test your implementation on the InvertedPendulum-v1 environment by running</li>

</ul>

With the given configuration file config.py, the average reward should reach 1000 within 100 iterations. <em>NOTE: Again, we only require that you reach 1000 sometime during training.</em>

Include the tensorboard plot for the average reward in your writeup.

Now, test your implementation on the InvertedPendulum-v1 environment without baseline by running

Include the tensorboard plot for the average reward. Do you notice any difference? Explain.

<ul>

 <li>(7 pts) (HalfCheetah-v1) Test your implementation on the HalfCheetah-v1 environment with <em>γ </em>= 0<em>.</em>9 by running</li>

</ul>

With the given configuration file config.py, the average reward should reach 200 within 100 iterations. <em>NOTE: There is some variance in training. You can run multiple times and report the best results or average. We have provided our results (average reward) averaged over 6 different random seed in figure </em><em>1 </em>Include the tensorboard plot for the average reward in your writeup.

Now, test your implementation on the HalfCheetah-v1 environment without baseline by running

Include the tensorboard plot for the average reward. Do you notice any difference? Explain.

Figure 1: Half Cheetah, averaged over 6 runs

<h1>2           Best Arm Identification in Multiarmed Bandit</h1>

In this problem we focus on the Bandit setting with rewards bounded in [0<em>,</em>1]. A Bandit problem instance is defined as an MDP with just one state and action set A. Since there is only one state, a “policy” consists of the choice of a single action: there are exactly <em>A </em>= |A| different deterministic policies. Your goal is to design a simple algorithm to identify a near-optimal arm with high probability.

Imagine we have <em>n </em>samples of a random variable <em>x</em>, {<em>x</em><sub>1</sub><em>,…,x<sub>n</sub></em>}. We recall Hoeffding’s inequality below, where <em>x </em>is the expected value of a random variable is the sample mean (under the assumption that the random variables are in the interval [0,1]), <em>n </em>is the number of samples and <em>δ &gt; </em>0 is a scalar:

Pr

Assuming that the rewards are bounded in [0<em>,</em>1], we propose this simple strategy: allocate an identical number of samples <em>n</em><sub>1 </sub>= <em>n</em><sub>2 </sub>= <em>… </em>= <em>n<sub>A </sub></em>= <em>n<sub>des </sub></em>to every action, compute the average reward (empirical payout) of each arm and return the action with the highest empirical payout argmax. The purpose of this exercise is to study the number of samples required to output an arm that is at least -optimal with high probability. Intuitively, as <em>n<sub>des </sub></em>increases the empirical payout converges to its expected value <em>r<sub>a </sub></em>for every action <em>a</em>, and so choosing the arm with the highest empirical payout <em>r</em>b<em><sub>a </sub></em>corresponds to approximately choosing the arm with the highest expected payout <em>r<sub>a</sub></em>.

<ul>

 <li>(15 pts) We start by defining a <em>good event</em>. Under this <em>good event</em>, the empirical payout of each arm is not too far from its expected value. Starting from Hoeffding inequality with <em>n<sub>des </sub></em>samples allocated to every action show that:</li>

</ul>

Pr

In other words, the <em>bad event </em>is that at least one arm has an empirical mean that differs significantly from its expected value and this has probability at most <em>Aδ</em>.

<ul>

 <li>(20 pts) After pulling each arm (action) <em>n<sub>des </sub></em>times our algorithm returns the arm with the highest empirical payout:</li>

</ul>

Notice that <em>a</em><sup>† </sup>is a random variable. Define <em>a<sup>? </sup></em>as the optimal arm (that yields the highest average reward <em>a<sup>? </sup></em>= <em>argmax<sub>a</sub>r<sub>a</sub></em>). Suppose that we want our algorithm to return at least an <em> </em>optimal arm with probability 1 − <em>δ</em><sup>0</sup>, as follows:

Pr<em>.</em>

How many samples are needed to ensure this? Express your result as a function of the number of actions <em>A</em>, the required precision <em> </em>and the failure probability <em>δ</em><sup>0</sup>.