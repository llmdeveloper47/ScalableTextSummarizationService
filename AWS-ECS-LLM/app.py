import openai
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain import OpenAI
import json
import boto3
from botocore.exceptions import ClientError


def get_secret():

    secret_name = "openai_key"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        
    except ClientError as e:
        
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    # Your code goes here.

    secret = json.loads(secret)

    return str(secret["OPENAI_API_KEY"])


#1. Generate a Summary of the Text
def prediction_pipeline(text):
    print("Inside prediction pipeline")
    text_splitter=CharacterTextSplitter(separator='\n',
                                        chunk_size=1000,
                                        chunk_overlap=20)
    text_chunks=text_splitter.split_text(text)
    print(len(text_chunks))

    llm = OpenAI(openai_api_key = get_secret())

    docs = [Document(page_content=t) for t in text_chunks]
    chain=load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)
    summary = chain.run(docs)
    
    return summary


user_input = st.text_area("Enter Text To summarize")
button = st.button("Generate Summary")
if user_input and button:
    summary = prediction_pipeline(user_input)
    st.write("Summary : ", summary)


# text1 = "Coco Gauff, the world No. 10 women’s singles player, has defeated Belarusian Aryna Sabalenka 2-6 6-3 6-2 with a dramatic comeback in the women’s US Open final.The star-studded crowd erupted with applause after Gauff’s home-turf victory at Arthur Ashe Stadium in Queens. The win is 19-year-old Gauff’s first career grand slam and makes her the first American teenager to win the US Open since 23-time major champion Serena Williams took the title in 1999.I feel like I’m in shock at this moment,” said an emotional Gauff after her win. “God puts you through tribulations and trials, and that makes this moment sweeter than I would have imagined.She thanked her family, her team, and “the people who didn’t believe in me.”Bidding for her second major title of the year, the soon-to-be women’s world No. 1 Sabalenka made quick work in the first set, breaking Gauff’s serve three times to win 6-2 in dominant fashion.However, with the packed crowd chanting “Let’s go Coco,” Gauff raised her level in the second set, going up a break before eventually taking it 6-3 to force a deciding third set.A locked-in Gauff took control in the third set, going up a double break to inch ever closer to her maiden grand slam title. Although Sabalenka took the next two games, Gauff closed out the match to become the 12th teenager in US Open history to win the title.“I don’t know, I just knew that if I didn’t give it my all, I had no shot at winning,” Gauff said on how she found the strength to rally after dropping the first set.In her run to the final, the athlete twice lost the first set of a match, once in the first round against Laura Siegemund and again in the third round against Elise Mertens.With the victory, Gauff becomes the third American teenager to win the US Open title, joining Williams and Tracy Austin. She is set to move up to No. 3 in the WTA singles rankings, and co-No. 1 in doubles along with compatriot Jessica Pegula.After clinching the victory, Gauff dropped to the ground before getting up to hug Sabalenka. Afterward, Gauff was overcome with emotion and knelt down to take in the moment.Gauff poked fun at her father after the match as she thanked her family. “Thank you first to my parents,” she said. “Today was the first time I’ve ever seen my dad cry. He doesn’t want me to tell y’all that, but he got caught in 4K!”Gauff also told reporters her parents helped when she would be too self-critical, placing too much value in whether she won or lost.“I used to put my tennis and compare it to like my self-worth. When I would lose, I would think, you know, I was not worth it as a person. So having my parents always remind me that they love me regardless of how I do helped me today.”When asked the significance of being the latest Black woman to win the women’s singles title, Gauff credited prior champions such as Venus Williams and Serena Williams, who “paved the way for me to be here” and added she was inspired by seeing Sloane Stephens win the US Open in 2017.“I hope that another girl can see this and believe that they can do it, and hopefully their name can be on this trophy, too,” she said.Meanwhile, despite the loss, the Belarusian star will move to No. 1 in the WTA singles rankings on Monday, ending Iga Świątek’s 75-consecutive week reign.Sabalenka congratulated her competitor, saying, “I hope we play in many more finals” and calling Gauff “amazing.”The American in turn congratulated Sabalenka on her rise to the No. 1 position. “Aryna is an incredible player,” she said. “Congratulations on the No. 1 ranking, it’s well deserved.”At a news conference after the match, Sabalenka said the loss was a “lesson” for her and she had started “overthinking” during the second set.“It’s me against me,” she said. Gauff “was moving really and defending better than anybody else.”“I was playing against the crowd,” she added.The last time Gauff and Sabalenka met was in the quarterfinals of Indian Wells in March, with the Belarusian winning comfortably, 6-4 6-0. Saturday’s final was an altogether different contest, however, with Gauff having improved rapidly in the six months that have passed since that defeat.The 19-year-old has won three WTA titles this season, including the biggest of her career in Cincinnati just before the US Open.The competition was the second grand slam final of Gauff’s career after reaching the French Open final in 2022, where she was swiftly defeated by Iga Świątek.Following her 6-4 7-5 semifinal win over Karolína Muchová, Gauff spoke about the improvement in her mentality, going from somebody blighted by imposter syndrome to now believing she is capable of contending with the best players in the world.She is not only contending but can now be regarded as one of the best players in the world after this win.Gauff was facing a formidable opponent – the best player in the world. Until her semifinal against Madison Keys, Sabalenka had been dominant in New York – not dropping a set and never losing more than five games in a match.However, despite defeat Sabalenka’s run to the final has capped a remarkable year in which she won three titles – including her first grand slam at the Australian Open and her sixth Masters 1000 title in Madrid."
# prediction_pipeline(text1)

text2 = "Representative Matt Gaetz, the far-right Republican from Florida, said on Sunday that he would move this week to oust Speaker Kevin McCarthy from his leadership post, promising to follow through on weeks of threats to try removing him for working with Democrats to keep the government funded.Mr. Gaetz’s announcement came the day after Mr. McCarthy, in a stunning reversal, steered around Republican opposition to a stopgap spending plan and turned to Democrats to help him push legislation through the House to avert a shutdown. Mr. McCarthy, a California Republican, said he knew he was putting his speakership at risk by doing so, and dared his detractors to make a move against him.In an interview on CNN’s “State of the Union” on Sunday, Mr. Gaetz, Mr. McCarthy’s main tormentor, said he would do just that. He said he would soon bring up a measure called a “motion to vacate,” which prompts a snap vote on whether to keep the speaker in his post.“I think we need to rip off the Band-Aid,” Mr. Gaetz said. “I think we need to move on with new leadership that can be trustworthy.”Mr. McCarthy shrugged off the threat, predicting that Mr. Gaetz’s effort to remove him would fail and was motivated by a petty grudge rather than a substantive dispute.“I’ll survive. You know this is personal with Matt,” he said during an interview on CBS’s “Face the Nation,” accusing Mr. Gaetz of being “more interested in securing TV interviews than doing something.”“He wanted to push us into a shutdown,” the speaker said.“So be it, bring it on. Let’s get over with it and let’s start governing,” Mr. McCarthy added. “If he’s upset because he tried to push us into a shutdown and I made sure the government didn’t shut down, then let’s have that fight.”Mr. Gaetz had long threatened to oust Mr. McCarthy for going back on several promises he made to Republican hard-liners to win their support to become speaker, including demands for deep spending cuts. In the interview, he accused Mr. McCarthy of lying to his G.O.P. members during spending negotiations and making a “secret deal” with Democrats concerning funding for Ukraine, which he and dozens of other conservative Republicans have opposed.“Nobody trusts Kevin McCarthy,” he added, predicting that the only way Mr. McCarthy would remain speaker by week’s end is “if Democrats bail him out.”Though most House Republicans still support keeping Mr. McCarthy on as speaker, Mr. Gaetz’s plans pose an existential threat to his tenure because of the slim majority the G.O.P. holds in the chamber. If Democrats were to vote against Mr. McCarthy — as is almost always the case when a speaker of the opposing party is being elected — Mr. Gaetz would need only a handful of Republicans to join the opposition to remove him from the post, which requires a simple majority vote.To avoid that fate, at least some Democrats would have to either vote to keep Mr. McCarthy in office, or simply skip the vote or vote “present” — neither for or against. That would lower the threshold for a majority and make it easier to defeat Mr. Gaetz’s motion.It is not clear whether Democrats would help Mr. McCarthy, however, particularly because he recently announced that he was opening an impeachment inquiry into President Biden despite lacking any evidence of wrongdoing. Most Democrats regard Mr. McCarthy as an untrustworthy figure who has spent months catering to the whims of his right wing. He has turned to Democrats only when his back is up against the wall, as he did in the spring to avoid a federal debt default and again on Saturday, during the waning hours of the fiscal year, to keep the government open.“I believe that it’s up to the Republican conference to determine their own leadership and deal with their own problems, but it’s not up to Democrats to save Republicans,” Representative Alexandria Ocasio-Cortez, Democrat of New York, said on “State of the Union.”Ms. Ocasio-Cortez said she would “absolutely” vote to remove Mr. McCarthy, calling him a weak leader who had lost control of the chamber’s Republicans and voicing skepticism that he could offer Democrats anything to gain their assistance.“I don’t think we give up votes for free,” she said.When asked whether Democrats should help protect Mr. McCarthy, Mr. Biden declined to weigh in. “I don’t have a vote on that matter,” he said on Sunday after delivering remarks about the stopgap bill. “I’ll leave that to the leadership of the House and the Senate.”Mr. McCarthy said in his “Face the Nation” interview that the House minority leader, Representative Hakeem Jeffries, Democrat of New York, had not indicated to him how he might vote on a motion to oust the speaker.It is also unclear how many Republicans Mr. Gaetz might rally to vote against Mr. McCarthy over the next few days. Representative Byron Donalds, Republican of Florida, who has criticized Mr. McCarthy but also clashed with Mr. Gaetz in recent weeks, said during an interview on “Fox News Sunday” that he had not decided how he would vote on a motion to vacate.“I think he is in trouble,” Mr. Donalds said of Mr. McCarthy, adding that he would “really have to think about” how he planned to vote.Still, Mr. Gaetz, expressed confidence that he would eventually rally enough votes among Democrats and Republicans to oust Mr. McCarthy as speaker, even if his opening attempt this week fails.“I might not have them the first time, but I might have them before the 15th ballot,” Mr. Gaetz said on ABC’s “This Week,” making a pointed reference to the number of attempts it took Mr. McCarthy to secure his speakership in January. He added, “I am relentless, and I will continue to pursue this objective.”Mr. Gaetz did not say who he would like to see replace Mr. McCarthy as speaker if he is deposed, arguing that it would be unfair to speculate while the House’s second-highest-ranking Republican, Representative Steve Scalise of Louisiana, is being treated for cancer.“I want to see how Steve Scalise comes out of that,” Mr. Gaetz said.That left open the possibility that the top post in the House could remain vacant for some time, with Mr. McCarthy forced out and nobody else able to muster the votes to replace him.The situation has left mainstream Republicans, including those in politically competitive districts who have toiled to distance themselves from their party’s extreme right, fuming.Representative Mike Lawler, Republican of New York, accused Mr. Gaetz of being “duplicitous” and engaging in a “diatribe of delusional thinking.”In an interview on ABC that aired just after Mr. Gaetz’s appearance, he accused the Florida Republican of breaking faith with the House G.O.P. and its rules by pushing ahead with the motion to vacate when a majority of the chamber’s Republicans did not share his animus against Mr. McCarthy. He also argued that the move would undermine all of the work Republicans had done to advance their conservative policy agenda.“This will all be torpedoed by one person who wants to put a motion to vacate for personal, political reasons,” Mr. Lawler said, noting, “We have to work together as a team.”"
prediction_pipeline(text2)


