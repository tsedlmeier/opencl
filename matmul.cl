__kernel void mat_mul(int widthA, int heightA, 
                      int widthB, int heightB, 
                      __global float* A, 
                      __global float* B, 
                      __global float* C)
{
// Get global indices of work-item
    int row = get_global_id(1);
    int col =get_global_id(0);
    
    // Compute single element of C
    float sum = 0.0f;
    for(int i=0; i<widthA; i++)
    {
        sum += A[row*widthA + i] * B[i*widthB + col];
    }
    C[row*widthB + col] = sum;
}
