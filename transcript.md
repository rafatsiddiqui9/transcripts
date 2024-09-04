Introduction 
there's elements of gdpr that clearly didn't work you know like they you need to accept cookies and everyone is like 
yeah yeah click click click click click yeah but what did work is actually not just data scientists but anyone in 
business at least started to think a little bit more like if we're Gathering data it's not just our data it's also 
our customers's data uh what is the purpose for which we are gathering this data did the customer give give consent 
for that and I think there will be a similar kind of Brussels effect of of the EU regulation here because um yeah 
like for starters doesn't matter if you're like an American company or a Brazilian company or whatever and you 
operating in um in Europe then you need to follow these [Music] 
rules all right everyone welcome to another episode of the twiml AI podcast I am your host charington and today I'm 
joined by Peter vanderpu Peter is director of the AI Lab at pegga systems and assistant professor at The leaden 
Institute of Advanced Computer Science at leaden University in the Netherlands 
before we get going be sure to take a moment to hit that subscribe button wherever you're listening to Today's Show Peter welcome to the podcast it's 
great to be here uh it's great to connect with you again we had an opportunity to collaborate actually 
quite a while ago on uh an AI program at Pega and uh we've been wanting to catch 
up and talk about some of the work that you've been doing on bias mitigation and in particular uh where academic fairness 
metrics tend to fall short in how to close those gaps for automated decision-making I'm looking forward to 
that conversation and I'd love to have you uh start us off by giving us a little bit about your 
background yeah awesome um a bit of an old timer in AI you I started to study 
AI in Believe It or Not 1989 and uh graduated from a master in the mid 90s 
um and uh like first did a lot of interesting work in uh let's say a 
commercial type research so uh with all kinds of different forms of data like 
structured data images video uh semantic search engines yet Etc so that was all 
in the in the '90s but then um some point I I I you know it's maybe the typical kind of uh uh data scientist 
let's say five years into your career I became a little bit yeah almost frustrated at all the interesting data 
sign stuff that you build you do a project you L your heels and then you come back later and even back then it 
was really yeah it was all on a project basis it was almost like uh the data science then evaporated and everyone 
went back to working uh working the old way so that's that's one I joined decided to join a startup which was more 
focused on decisioning so as a way to make ai ai actionable you know how can 
you how you can make automated decisions in marketing and service and loan applications claims um and because then 
yeah the focus is really on on embedding uh embedding AI into a process not just 
I'm an AI machine learning guy but not just the AI machine learning but even the good oldfashioned uh 
AI you know so that's also where then ultimately ended up at at Pega you know in the interim a 
bunch of new Hypes have happened deep learning around 2010 generative AI now but it's been 
quite a journey yeah exactly but that whole topic of how to make AI actionable 
but also to create make sure it creates responsible impact yeah that that's been 
kind of a bit of a Trail throughout mhm and one of the interesting points of 
European AI Act 
context for that that has been particularly important for you uh being 
in Europe is some of the recent regulation that's coming down the line 
through the European AI act can you introduce us to to that and tell us what 
you see there yeah know absolutely um so um yeah of course we've been haggling 
about that EU style for like almost six years 2018 it started uh with a 
so-called high level expert group on AI and they came together try to Define you 
know what kind of definition of AI should we pick but also what are some of the key ethical principles like 
transparency accountability robustness but also fairness and that then evolved 
into kind of a proposal for uh for an eui act um already a couple of years ago 
and that has been under negotiation for a couple of years but the final text got uh accepted by uh Parliament European 
Parliament in March and now there's only one one final hurdle which is pretty much only a 
formality uh uh yeah to to to to get that legislation into effect and and the 
key IDE of the eui act and so it's it's covering these ethical principles that I 
spoke about but higher level it looks at also uh takes a risk based approach 
right so it it looks at an AI system and not not AI in general that would not 
make sense you know you can use it for good things you can use it for bad things you so you need to look at a particular AI system a particular use 
case and then uh look at uh what is the risk to do harm yeah and uh there are 
certain things that that are not allowed it's a very small list like whatever uh um um social credit scoring like you 
have in China for example um uh but then there's high-risk AI systems and let's 
say regular AI systems and the higher risk AI systems uh for instance uh that 
that's yeah come on more scrutiny I'm curious how it even defines an AI system 
what I really liked about um the eii ACT is that it you know people didn't start 
haggling about what is the difference between machine learning and statistics and uh and and things like that it it 
was taking the point of view of let's say let's say the uh average consumer or 
citizen who gets to gets exposed to automa decision- making yeah from that 
perspective it doesn't really matter if if you're being whatever denied Alan or investigated for fraud or whatever it is 
doesn't matter that much whether that's through a machine learning model or through a regression model or through 
even a combination of models and rules good oldfashioned AI in in all of those 
cases uh it's it's an automated decision in in an AI system so from the start 
actually that that definition on purpose the technical definition was kept quite 
broad because even purely technically mathematically there is no fundamental 
difference between let's say a small neural network or a regression model yeah if you have a neural network 
without the hidden layer linear transfer function that's that's your regression model right but but it was good that 
they took that that that kind of broad approach and then uh look more at the use of of that AI in a particular system 
and the current definition is a definition from the oecd that that has been uh accepted and actually you know 
it got me even more excited because it's using a definition which very much looks like a field that almost predated AI 
which is cybernetics right so good old Norbert Wier it because it's talking 
about AI systems that that that indeed are driving automated decisioning in an 
environment can have an impact on the on that environment as well yeah so it's 
really taking this um it's really trying to move away from let's say the pure 
technical definition of what algorithms do you use towards more like what is the use of an AI system as as let's say a 
real-time decisioning system or a real time or or a generative system that's operating in environment that that's 
interacting with users that has an impact on the environment and so is the act of making 
the decision or making a decision that impacts the users of the system is that kind of the the 
key um element that defines the scope of this act or is it uh broader than that 
uh it's um if I get your question correctly it's 
like the highest level apart from these the general level principle principles of transparency and fairness and 
accuracy and robustness and safety and privacy the higher level thing is what is the objective of my AI system yeah is 
that a good objective because certain objectives are forbidden and other 
objectives for these other objectives is what's the risk to do harm as they they put it right so and then you have these 
higher risk systems that you need to put on there like higher levels of of of scrutiny yeah uh like like you know 
providing access to financial services that that that that is actually uh 
providing or blocking access to for example is a such an example or or providing access 
or or blocking access to education or or whatever it is yeah so um but yeah so 
but it's really in that sense um not some theoretical law it really looks at like oh you know what do you use it for 
you know like what is the risk to do harm and that's it's a very I think sensible proo is risk in this 
context uh defined abstractly and uh it's up to the 
implementor to um figure out if their system is risky or not or are do they 
explicitly list um you know these specific uses are considered high risk 
medium risk low risk it seems like it would be very difficult to uh catalog 
every possible use case and assign it a risk it's it's M it's a mix so uh yes 
indeed they they leave a lot of it to yeah ultimately ultimately to uh I don't 
know certain cases would be brought to court or whatever um uh to to Really 
fill in fill in the details yeah uh of what is what is um you 
know what what constitutes a high level of risk or or or you know if there are 
certain requirements around whatever transparency or whatever did you actually meet those uh those 
requirements but what what they did do is list uh a bunch of things that for 
sure fall under uh forbidden use and for sure fall under high 
risk and now here in the US one of the 
EU AI Act vs other AI Regulation Acts 
ideas that's coming up around AI acts and AI safety or um constraints around 
training large language models if it's going to be lger than so and so parameters or require more than so and 
so uh compute um you know those are forbidden or require you know some 
degree of Regulation does the European AI act get into that type of Regulation 
or is it primarily focused on consumer protection or um you know citizen 
protection uh in terms of decisions and interactions yeah no it does actually 
get into it honestly the the the history of these type of law this type of law in in 
Europe is a little bit from a product protection point of view right so and that still shines through a little bit 
when you bring bring a product to the market well with AI you don't bring a product to the market you know like uh 
you're you're you know many of the whatever if you're a bank and insurance company whatever you're creating your 
own models right so so it's slightly different than bringing a product on the market but um they did actually uh 
address these points around generative AI yeah because when I said like in the beginning technology can be good or 
bad uh nor is it neutral we need to think about an AI system that was 
actually very kind of convenient position a couple of years ago but then of course the ni came around the corner 
and and this this whole concept of need the foundation models that everyone can reuse and then uh it was like oh you 
know like should be completely is should we treat it as a completely different 
type of thing or should we apply exactly the same rules and you know there's a little bit of a Midway solution to be 
honest I I do think it's healthy to not just jump to the conclusion you need to do something completely different I 
think the the the the general framework they haven't changed the general framework and it still holds but they 
did make like additional kind of proposals around how to deal with the foundation models because Foundation 
models they're a little bit in between this General technology and an actual AI 
system or product because you you can say it's generic so it's not a system and it's ultimately people that use the 
model in a particular context yeah that that's that's right but on the flip side people cannot typically don't change the 
foundation model so it's almost like a half fabricate right so and that's where indeed you know like 
there's additional uh requirements that were were put in also taking into account the the relative size of the of 
the model you know like how how complex or large it is um um I forgot the exact 
requirement but it's a it has to do with how many paa flops uh to train the model or something like that yeah so it's not 
not parameters but but more actual kind of uh uh uh training training cost I 
believe yeah so that's that's similar in that sense as in the um as in the US of 
course in the US you also have I think the policy level 
there's frankly yeah there's a lot of commonality across different regions right so even in China where they were 
first to put out uh AI law of course there's maybe elements in that that 
don't resonate maybe more with uh values in the US or in Europe like uh how 
closely they monitor their their citizens for example a little bit too close maybe but on the flip side there's 
also very like all these elements of transparency fairness Etc they're also part of that the Chinese legislation um 
likewise the thinking when you had the um the Biden exact order on um on AI 
last fall uh or um uh uh previously there was also a 
policy document coming out of the white house uh around AI yeah it has these similar concepts of you know the general 
Bas layer of of of things of of these ethical principles but also what is the 
type of use it's just not legislation you know like uh I think at this stage you know uh that that was just I think 
the political climate is not ready for that at the moment in the in the US with elections coming up and things like that 
but in that sense the White House did take like a prudent approach uh also by 
messaging more to government different government agencies and departments this is how we want you to use AI but also 
how want used to uh uh you to use AI systems yeah and similar how does a 
trickle down effect the Brussels effect from EO legislation the these type of 
policies that are directed at uh US government agencies they do trickle down into U into the rest of the uh into the 
rest of the economy right yeah and when I think about uh European regulation um 
Impacts of EU AI Act 
you know clearly there's lots of European regulations but you know when I think about it and its application in 
the tech Bas I can't help but think of gdpr which um you know I just remember 
having a lot of conversations at the time with folks about how it would impact the way data scientists and 
practitioners uh approach privacy and um 
you know it had Global impact uh for organizations do you expect the EU AI 
act to have uh a similarly broad footprint or uh you know are they talking about 
things that people are already doing um and less about prescriptive things that 
people need to do yeah no I think Al like I think it's I it's a great remark 
it's always good to look back to some other forms of Regulation how did how how did that go right so uh from a gdpr 
perspective if if I well this is a little bit of an opinionated comment right so but just to take take a 
position and then there's elements of gdpr that clearly didn't work you know like they you need to accept cookies and 
everyone's like yeah yeah click click click click yeah just just give me give me my stuff right so that that didn't 
work yeah but what did work is actually exactly what you say and that 
data scientist not just data scientist but anyone in business at least started to think a little bit more like if for 
Gathering data it's not just our data it's also our customers data uh what is 
the purpose for which we are gathering this data did we kind of kind of give some level of consent or did did the 
customer give give consent for that so I think actually the the yeah the biggest impact is almost like self um 
self-regulation in a good way right where not self-regulation but um how do you call it that you adapt your your 
behavior in a way I think that's actually the biggest impact not so much the big cases that were brought forward 
and where people uh where companies or public organization need to pick pay pay 
big fins what I hear you saying he you know I hear you saying that the big 
impact is establishing standards of care more so than uh enforcement regime 
that's going to come and police the market yeah and I think that worked and I think there will be a similar kind of 
Brussels effect of of the EU where regulation here because um yeah like for starters doesn't matter if you're like 
an American company or a Brazilian company or whatever and you operating in um in Europe then you need to follow 
these rules yeah it doesn't matter whether you're American Brazilian or Chinese um and then if you're a global 
company uh then yeah yeah you're kind of going for the most common denominator anyway right we don't see that entirely 
yeah there's similar legislation like the digital Service Act in the EU and then you can see that whatever uh uh the 
likes of meta and and Google and they release certain things later in the EU than uh in other areas but yeah the 
emphasis here is also a little bit on later they're releasing it they do they do Rel release it and then they if if 
they make sure that they take the boxes and put in the uh the extra safeguards right so but I think um like at Pega we 
are a big fan of uh I'm personally a big fan of ethical use trustworthy use of AI a Pega we are that as well which is 
maybe not what you would expect from a big Tech uh uh vendor uh but but we we really support 
it because I don't think regardless of whether you have regulation or not I don't think there's any sustainable 
future for irresponsible use of AI That's a very simple way to think about 
it right so and it may yeah it may work in the short run but in the long run customers will vote with their feet yeah 
they will take the business elsewhere yeah so so I think in that sense um 
that's Karma aside you know Karma wise it's important yeah but but even 
practically you know like uh to use a fancy word from utilitarian perspective 
ethical perspective uh forget principles and Karma it's not going to fly you know 
like bad bad uses of AI that might fly fly in the in the short term but it will 
never fly in the long term people will just not accept the technology in the long run anded take their business 
elsewhere right so I think uh so yeah I think it's very healthy if if you great 
a Level Playing Field in that sense uh in the the lead into the conversation I referenced um you know 
Real-world application challenges of metrics in the EU AI Act 
some of the fairness and bias metrics that organizations have started looking 
at to uh understand the impacts of the way that they're applying uh machine 
learning does the ACT specify metrics like that is it 
prescriptive to that level of detail right yeah no exactly it it specifically 
doesn't tell you what how you should do it but it tells you you should do it and 
specifically for high-risk applications uh you should you should do it yeah so 
you need to prove that that you took uh that you well uh that you at least made 
a conscious effort to um to look at fairness and uh to look at fairness and 
bias MH and so one of the challenges that you point out is that the metrics that are often talked about uh don't 
really work in the real world scenarios that that you see with Enterprises can you give us some examples of where the 
disconnect lies yeah like I think the disconnect because I'm I'm also an academic and I 
love doing uh re research but sometimes did it not just for fairness and bi but 
in general for many fields and then there's a grand idea about the field and then everyone is kind of focusing on 
some subtopic where it's extremely crowded yeah and and I think what you see a lot is in in in uh know this is a 
gross generalization so apologize for anyone who's not doing this but but 
there is a lot of focus on yet another fairness metric to measure uh by high in 
a machine learning model at design time yeah so and you hear me or you another sweet or Ensemble of fairness metrics to 
yeah choose from yeah yeah and I'm not saying that these fairness metric metrics are not important but to some 
degree there's other you know there's also not bigger fish but at least other fish to fry as well 
so well you know like if if I say uh we measure fairness in a model like going 
back to that regular if I don't get my loan right so that's not just based on one model you know 
that's a you know probably default model maybe a fraud model but also a whole bunch of rules that go into the mix to 
uh decide whether you get a loan or not and um yeah the the LA yeah for instance 
UA but it's similar with uh uh with uh 
lending legislation in the in the US the point is do you get the loan or not yeah not like uh uh just looking at 
one particular model in a set of models and uh and rules so you do need to be able to look at a different level not 
just at the level of a model but at the level of a full automated decision which is a combination of uh of models uh 
rules uh data going into uh these models and and rules yeah um I mean does that 
is that in what ways is that technically interesting when I think about the bias metrics that are applied to models it's 
typically you are kind of profiling a data set before and after or d a data 
set before and a decision after and trying to make inferences between the two um it seems like in the case you're 
describing you can just uh create a larger or broader definition of model 
and assume that the model is the whole decision uh in other words a lot of these fairness metrics don't really 
apply to you know they're not uh kind of intrinsic properties of the model it's 
more about the data and the decision yeah but that I mean that's a good thing that's an opportunity it means that a 
lot of your fairness metrics they they if you have disperate impact or rate 
Ratio or gen index or like we use those for example they acquire equally at 
let's say at the decision level than at the model level of course where it becomes interesting then is um is uh um 
where it becomes algorithmically interesting is uh well how do you break that down you know like if you have an 
enre decision that consists of many rules and logic like how you know how do you trace fairness through uh such a 
logic for example or uh what's also interesting is um uh maybe looking at 
well you're you're going to you might be generating a lot of different alerts like uh I don't know our and this is by 
the way largely an unsolved problem and we um uh I don't know some of our 
customers are using uh uh decisioning real time decisioning in one to one personalization and then you're talking 
like deciding across a library of 3,000 uh next section recommendations 
with a model behind each of those and then tons of rules on top and waiting uh 
and whatnot yeah so if if you you're you're just by the large law of large 
numbers you're B bound to yeah to find certain levels of biases uh in those models so how do you 
make sure that um yeah you're not creating a downstream Problem by saying 
oh we have our metrics we run our simulation great fantastic and then you spit out thousands of alerts where you 
go like what do I need to do now yeah so and I think that problem gets AGG 
aggravated which is great for research we like bigger problems if you don't just look at design time but why should 
you only look at design time you also need to look at runtime as in when uh decisions are being made or maybe at all 
the time a post decision you want to have the ability to replay maybe 
different versions of that um of that logic so those are all interesting kind 
of still pretty much open-ended research problems um that um uh moving from 
design time to runtime monitoring of fairness for example or uh how to deal with all these alerts that are are being 
generated or or as data scientists we have a tendency to you know when you 
have have a hammer everything looks like a nail so you go like oh if I detect some bias oh I'm just going to create a 
bias correction algorithm that automatically uh corrects for that bi yeah that's of course as a data 
scientist what we do but that's it's not always the the right thing to do because you could be covering up the root cause 
of the bias yeah so uh there's some great examples uh of 
that in uh like um in in in in healthcare in the US um where you know like um where it's 
important to get into the yeah into the root cause of uh what what's behind the thing that creditor Buys so these are 
just a random sprinkling uh of of yeah research problems yeah and 
implementation problems which are are are not typically not not well they 
could deserve more attention I think in um in in research and and also yeah and 
that's not um to do even more research but I think from a practical perspective it's really important I think these are 
the real world problems that people who are implementing yeah AI 
systems in the real world are having to deal with and so from a practical perspective 
Addressing the challenges in fairness and bias metrics 
like what do you say to someone who has to deal with these problems how do they go about trying to address 
them yeah um now like so and when people 
hear compliance they go like oh we need to be compliant everywhere yeah and of course you need to be compliant 
everywhere but it does mean also that you do need to uh pick your battles and 
you do need to look at like okay but where could we where could some real harm uh be created right so I think that 
would would lead you towards focusing uh on the right erors for 
example uh if I have um whatever um let's say I'm a bank and I'm selling 
credit cards and Loans uh yeah I can yes I can can look at my marketing models yes you want to have some level of uh 
fairness checking in there but the models that determine your um yeah your 
limit on your credit card that's already more uh more interesting there was this 
case of U uh someone using an apple card and claiming like uh hey uh that's 
strange my uh uh the limit on my credit card is um 20 times higher than they 
limited my wife gets right so uh what what's causing this and then some other 
person retweet that and this was about Apple card and it was Steve wnc right so 
I have the same thing the original was David Han Hansen this was shortly after the Apple card came out yeah yeah yeah 
and they investigated later on turned out not to they couldn't confirm there was a real issue at just to just to uh 
set the record straight on that one yeah but but but like do Focus you so in this 
example I started to say Hey you sell credit cards and Loans yes you need to look at your marketing decisions but uh 
the decisions that set your limits or the the decisions that decide whether you can have a credit card uh or not 
yeah or a loan or not or whether you're going to we're going after you for fraud 
H those are of course uh even more material decisions and so I think it's 
healthy in that sense also to uh well also take to take this risk based approach and see like hey where where 
where you know where do we need to prevent where can we prevent damage from happening yeah so uh that's something 
that data scientists are kind of uh yeah if you don't think too binary then uh 
you're actually quite open to it because you can yeah it's it's a little bit of a quantitative approach you say well where 
you know where do we expect um uh the largest risk and let's let's focus on 
there and then and then build it all out but um yeah I think that would be uh 
that will be a good approach so what sorry what specifically 
is the approach you're suggesting well the approach is to to well look at this risk to do harm for example and then 
focus on those decisions first uh uh like if I need to prioritize 
looking at Marketing in particular loan versus uh providing actually access to the loan yeah can you get can you get 
the loan or not the loan acceptance the loan acceptance is more important than a marketing I'm not saying there should be 
that we should reduce a bi in marketing but there is a difference here right so 
I think that could help Focus uh uh where you want to look 
um also there should be kind of a a culture uh 
um uh it should be embedded in in in the yeah at risk of sounding fluffy yeah but 
in the culture of the uh organization and the organization processes as well 
yeah so if I need to do that next to my day job and then the moment I find a problem I'm the one causing trouble yeah 
that's not a culture where where issues are being found and fixed yeah so um 
whereas if it's a culture like we recognizing there's Biers in these decisions there's no way you can 
eradicate buyers completely there is no way yeah you reduce buyers for one group 
it may impact another and but but we want to keep it within certain set 
levels and it's okay to find areas where you know models have drifted or rules 
have changed or the population has changed and where we're getting more into the uh in into the from the green 
zone into the orange Zone it's okay to find to find such an issue and and solve it that's a good thing yeah so I think 
that's uh um yeah I think that's that's also important uh to uh so the first 
thing is like find the areas where you can cause most harm is that's like your 
strategic Focus uh second is is indeed more from a governance operational point 
of view there needs to be be this culture that it's it's okay to fix these things and there's a recognition that 
it's fine to find issues because at least we can fix them and there will always be some level of biys and then 
it's more into the methods yeah with the methods think broader than just a model you know think about autom make 
decisions that are models and rules think broader than just doing lots of testing at design time and then you 
launch your models and logic and you never you never check them again yeah that's wrong you need to monitor at run 
time as well right so um think a little bit out out of the box in that 
sense MH and we started talking earlier 
Closing the gap in fairness metrics and real-world AI applications 
about um kind of this disconnect between the 
fairness metrics and real world application um does that disconnect get 
closed by um you know a looking at systems in 
their entirety and not just models and and be by looking at runtime in addition 
to uh design time or are there other gaps that uh teams need to be looking 
at yeah like H so um I I I do think that um it's it's 
interesting for me to have one leg in industry and one leg in academics if I put my academics hat on uh we we suffer 
from Le like I'm doing all this call I apologize let me explain this to you one 
more time because you you probably are not getting it yeah but if you flip it around I think if you almost like a an 
anthropologist would do field research if you would do AI ethics rather than talking about it you find out what the 
real issues are yeah so I I do think that that actually by um getting into 
the field yeah uh only then you will find out uh 
the bigger issues and these bigger issues sometimes they have these things like oh culture organization blah blah 
blah but sometimes they're actually a nice data science research puzzle yeah so like how to deal with tons of alerts 
that's a nice data science research puzzle or how could I is there a way to 
continue to check for fairness Biers continuously nice research puzzle yeah 
so or data science puzzle yeah so um uh so I do think it's it's it's healthy 
there's other aspects as well I I don't think we need even more fairness metrics you know like uh we have enough of them 
there's some interesting work around multiattribute uh fairness right so I 
think that's that's potentially interesting and so if you combine multi protected fields and not not just gender 
but combination of gender and age right so young females uh middle-aged 
middle-aged middle-aged men yeah uh are they unfairly treated 
um yeah but um yeah I I do think that that uh uh get getting 
into uh yeah getting into practice will lead to uh to interesting uh research 
problems and even yeah so we launched our our own 
ethical bias check we launched at uh like 20 what was it 2020 something like 
that I believe and so so um and then we thought like okay you know then we'll 
um um we we didn't want to over engineer it but take it out in the field and then 
and then then start improving it but uh even getting people to adopt that yeah 
and then see how can I build it into my working practice that's going I I 
expected that that would go a lot faster right so so in that sense I think it's also I'm not unhappy that um you know 
we're about to have the signature on the uh AI act because people will get a little bit more serious about these uh 
these uh these topics so you think the lack of adoption there is primarily 
Reasons for lack of adoption 
motivated by lack of incentive it's a little bit lack of incentive it's also a little bit like 
that people individually subscribe to that cause but of course the organization needs to do that as well 
yeah and then you know like as usual you know how how do you operationalize this 
that's that's the biggest uh the biggest tle yeah so um it is something that also indeed 
more in my academic research we we intend to do a a project on uh bias and 
fairness uh and and and actually also I think you can't blame the data scientist 
because this for instance this tradeoff between when do we offer for a loan or not you know thresholds and things like 
that that's not something um the data scientist is 
responsible for it's someone who owns whatever the lending product yeah sure 
Underwriters the risk manager or the risk ins yeah so those people need to 
know about get some level of understanding about what buy and fairness is right so one of the research 
project we want to do is to well first make an Outreach to that or audience and trying to explain in in in terms that 
resonate with that audience what bi and fairness is but also then take back like okay but what are the real problems that 
these problem that these people face and when they want to actually implement this and which I'm sure uh will stumble 
across uh many other things than just kind of have yet another fairness 
metric because uh whatever desperate impact is not working for me you know like that's that's not going to be the 
focus of the feedback that we're going to get from these people so is your 
Recap 
sense of the kind of the maturity of the market that most organizations 
aren't kind of thinking seriously about these issues and don't have a process in place to apply um you know some kind of 
you know rigorous assessment of models and and fairness to the products they're 
making um it's it's at risk of of giving you a complex answer or I'm not dodging 
the question here but I think it's a mix I think individual people actually you know have the right intent yeah um maybe 
outside of let's say data science analytics uh people are less accustomed with the whole topic right so that's why 
I do think it's important to explain it in in terms that a product manager or whatever uh a business line owner or or 
p&l owner can actually understand this uh this topic so the appetite is there I 
think um uh but I do think firms are only are are just at the start of 
actually uh operationalizing this right so there's may be very specific um uh 
Pockets where they're further ahead and then you should think of things like credit RK decisioning or 
or fraud or well maybe fraud a little bit less so but definitely crit with 
decisioning the the the the traditional Model Management type uh model board 
areas within financial institutions for example but uh I think the individual 
appetite is there the system appetite at the level of the organization that needs to that needs to get like a further push 
um I do see more and more clients getting uh yeah getting a little bit 
more uh into it I think uh another positive note is that um um yeah of 
course for AI altimers like me when Gen AI came I was you know is everyone is pretending 
that AI was invented in November what is it 200 22 when CH gpg came out come on um 
it was hard spell before yeah like but but of course there is a positive impact 
because uh gen is um quite different in 
the let's say different in the audience of who is working with the AI technology 
it was very much limited to the pure data scientist and the the other the decisioning AI that we have right so and 
with geni uh like even though consumers they have been exposed to AI like times a day 
for 20 years already with Google or listening to Spotify or whatever planning your at uh I think generative I 
is the first time that there was a wide audience very wide audience of people that got to make AI you know like uh 
even though we as data scientists don't think if you're writing a prompt for us it doesn't count as making AI but for 
many people it was their first experience of getting into actually creating some creating some Ai and that 
has led to a lot more attention for for AI and machine learning in U in uh 
companies and public organizations and yeah that that's where also immediately these questions then come up you know 
it's not just about hallucination and toxicity or filtering private data but 
it's also about um yeah uh yeah topics like 
fairness awesome well Peter thanks so much for joining us and taking a Time 
share a bit of your perspective on uh the way organizations are grappling with 
uh looming regulation like the European AI act yeah it was great to be 
here all right thank you thank you 
[Music] 
[Music] 
[Music] 
 - Generated with https://kome.ai
