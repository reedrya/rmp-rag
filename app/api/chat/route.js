import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import Groq from "groq-sdk";
import OpenAI from "openai"

const systemPrompt = 
`
You are a helpful and knowledgeable assistant for students using a "Rate My Professor" platform. Your task is to assist students in finding the best professors according to their specific queries, using the latest information available. For each student query, you must:
Understand the Query: Accurately interpret the student's question, identifying the key requirements such as subject, teaching style, difficulty level, or other specific criteria they might mention.
Retrieve Relevant Information: Utilize the RAG approach to retrieve the most relevant data from your knowledge base, ensuring that you consider a wide range of factors including student reviews, ratings, subject expertise, and any other relevant metrics.
Generate and Rank Responses: Rank the top 3 professors who best match the student's criteria. Each professor should be accompanied by a brief explanation of why they are a good fit based on the student's query.
Communicate Clearly: Present your findings in a clear and concise manner, making sure to highlight the key attributes of each professor, such as their strengths, teaching style, and overall rating.
Be Neutral and Helpful: Maintain a neutral tone, focusing on providing unbiased and helpful information to guide the student's decision-making process.
`

export async function POST(req) {
    const data = await req.json()

    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const groq = new Groq()

    const text = data[data.length - 1].content

    // getting errors here since Groq doesn't have an embedding API, used OpenAI for now as placeholder
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float'
    })

    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    })

    let resultString = '\n\nReturned results from vector db (done automatically):'
    results.matches.forEach((match) => {
        resultString+= `\n
        Professor: ${match.id}
        Review: ${match.metadata.review}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

    const completion = await groq.chat.completions.create({
        messages: [
            { role: 'system', content: systemPrompt },
            ...lastDataWithoutLastMessage,
            { role: 'user', content: lastMessageContent },
        ],
        model: 'llama3-8b-8192',
        stream: true,
    })

    const stream = ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content){
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            }
            catch (err) {
                controller.error(err)
            } finally {
                controller.close()
            }
        },
    })

    return new NextResponse(stream)
}