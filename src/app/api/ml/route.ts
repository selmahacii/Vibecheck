import { NextRequest, NextResponse } from 'next/server';

// Python ML service runs on port 5000
const ML_SERVICE_PORT = 5000;
const ML_SERVICE_URL = `http://localhost:${ML_SERVICE_PORT}`;

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const path = searchParams.get('path') || 'health';
  
  try {
    const response = await fetch(`${ML_SERVICE_URL}/${path}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('ML service error:', error);
    return NextResponse.json(
      { error: 'ML service unavailable', message: 'Make sure the Python server is running on port 5000' },
      { status: 503 }
    );
  }
}

export async function POST(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const path = searchParams.get('path') || 'analyze';
  
  try {
    const body = await request.json();
    
    const response = await fetch(`${ML_SERVICE_URL}/${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('ML service error:', error);
    return NextResponse.json(
      { error: 'ML service unavailable', message: 'Make sure the Python server is running on port 5000' },
      { status: 503 }
    );
  }
}
