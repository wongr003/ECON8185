function InterpolateEGM(cbar_vec, abar_vec, a1, ϵ, a1_size)
    a1_index = searchsortedlast(abar_vec, a1);

    ##Adjust indices if assets fall out of bounds
    (a1_index > 0 && a1_index < a1_size) ? a1_index = a1_index : 
        (a1_index == a1_size) ? a1_index = a1_size-1 : 
            a1_index = 1 
            
    c_low, c_high = cbar_vec[a1_index], cbar_vec[a1_index+1];
    a_low, a_high = abar_vec[a1_index], abar_vec[a1_index+1];
    c = (c_low*(a_high-x) + c_high*(a1-a_low))/(a_high-a_low);

    return c
end
