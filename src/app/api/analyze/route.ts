import { NextResponse } from 'next/server';

export async function POST(req: Request) {
    try {
        const formData = await req.formData();
        const image = formData.get('image') as File | null;
        const message = formData.get('message') as string;

        // 1. NLP Pipeline: Preprocessing
        const sanitizedMessage = message.trim();

        // 2. Try to connect to Local AI Kernel (Python/FastAPI)
        try {
            // Hybrid Cloud Support: Use remote URL if set, otherwise try the hardcoded tunnel or localhost
            // NOTE: This Hardcoded URL enables the Cloud Bridge without Vercel Env Vars
            const currentTunnel = 'https://rooted-joylessly-audry.ngrok-free.dev';
            const baseUrl = process.env.AI_BACKEND_URL || currentTunnel;
            const aiServerUrl = `${baseUrl}/analyze`;

            const aiFormData = new FormData();
            aiFormData.append('message', sanitizedMessage);

            if (image) {
                // Determine blob/file handling.
                aiFormData.append('image', image, image.name); // Pass filename
            }

            // Setup 5-minute timeout for large model downloads
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 300000); // 300s = 5 mins

            const aiRes = await fetch(aiServerUrl, {
                method: 'POST',
                body: aiFormData,
                signal: controller.signal,
                headers: {
                    'ngrok-skip-browser-warning': 'true'
                }
                // Do not set Content-Type header manually for FormData, fetch handles boundary
            });
            clearTimeout(timeoutId);

            if (aiRes.ok) {
                const aiData = await aiRes.json();
                return NextResponse.json({ reply: aiData.reply });
            } else {
                console.warn("Local AI Server returned error", aiRes.status);
                const errorText = await aiRes.text(); // Get error details
                console.warn("Error details:", errorText);
            }
        } catch (connectionError) {
            console.warn("Could not connect to Local AI Server (is it running?):", connectionError);
            // Fallback to Mock Response if Python server is down
        }

        // --- Fallback Mock Inference (UI Demo) ---
        await new Promise(resolve => setTimeout(resolve, 800));

        let reply = "I can help you with that.";
        if (image) {
            reply = `[Server Offline] I see an image named "${image.name}", but the Vision Engine is not reachable. Please start the backend server.`;
        } else {
            reply = `[Server Offline] The AI Brain is currently sleeping. Please run the backend server to wake it up.`;
        }

        return NextResponse.json({ reply });
    } catch (error) {
        console.error('AI Processing Error:', error);
        return NextResponse.json({ error: 'Inference failed' }, { status: 500 });
    }
}
