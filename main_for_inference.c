/*

************************* TERMS OF USE *************************

Users within the open community are fully permitted and encouraged to access, download, analyze, and use this software code
as long as proper credit is given to the authors in the citation below. The present material is released under the 
Attribution 4.0 International (CC BY 4.0) license.
    © 2024 Norwegian University of Science and Technology (NTNU) in Trondheim, Norway. All rights reserved.
    - by Jon Alvarez Justo

Citation: 
  Article's title: In-Orbit Deployment of 1D-CNN for Hyperspectral Image Segmentation for Optimal Satellite Operations
  Article's authors: Jon Alvarez Justo, Dennis D. Langer, Simen Berg, Jens Nieke, 
                    Radu Tudor Ionescu, Per Gunnar Kjeldsberg, and Tor Arne Johansen
  Year: 2024


****************************************************************

*/ 




// LIBRARIES
#include <iostream>
#include <cstdint> // Use of uint16
#include <cmath>   // For exponentials
#include <string>  // Convert number into ""
#include <stdio.h>
#include <vector> //  To work with vectors instead of arrays
#include <omp.h> // For OpenMultiProcessing pragmas

// Using std
using std::cout;
using std::endl;
using std::exp;
using std::to_string;

// MACROS (for pre-compilation)
// We try to maximize the pre-compiled variables to reduce calculations during run-time


// Metadata of a .bip capture
#ifndef D_LINES
#define D_LINES 956
#endif
#ifndef D_SAMPLES
#define D_SAMPLES 684
#endif
#define D_CHANNELS 120 
#define D_CHANNELS_REDUCED 112
    // E.g.: D_SAMPLES stands for dimensions for samples
    //       The D is added to distinguish that it refers to dimensions and not the total number of samples in a cube
#define BYTE_DEPTH_PER_SAMPLE_IN_CUBE 2 // 2 bytes for 16-bit depth: sizeof(uint16_t)
#define LINE_BUFFER_SIZE_IN_NUMBER_OF_SAMPLES D_CHANNELS * D_SAMPLES 
    // E.g.: 
    // One line has 684 samples for 120 channels, then it means that there will be 684 * 120 = 82 080 samples within one line to be read

// Number of bytes in float (needed for reading the model's parameters)
#define BYTE_DEPTH_OF_FLOAT 4 // 4 bytes for 32-bits float: sizeof(float)


#define CHANNELS_TO_BE_REMOVED_IN_RAW_DATA {10, 11, 12, 13, 116, 117, 118, 119}


// INPUT
#define INPUT_FEATURES 112

// LEVEL 1 -> CONVOLUTION
#define KERNEL_SIZE 6 // For convolution
#define NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER 6
#define SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER (INPUT_FEATURES - (KERNEL_SIZE - 1))

#define WINDOW_LENGTH_FOR_POOLING 2 // This applies to all the pooling layers in the CNN

// LEVEL 1 -> POOLING
#define SEQUENCE_LENGTH_INPUT_POOLINGLAYER1 SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER
#define NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER1 NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER // This depends on the number of filters from the previous convolutional layer
#define NUMBER_OF_WINDOWS_POOLINGLAYER1 (SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER / WINDOW_LENGTH_FOR_POOLING)
// Length of the pooled sequences
// This should truncate already the fractional part - the number of windows must be a floor since the last samples
// are discarded

// LEVEL 2 -> CONVOLUTION
#define SEQUENCE_LENGTH_INPUT_SECOND_CONV_LAYER NUMBER_OF_WINDOWS_POOLINGLAYER1
#define SEQUENCE_LENGTH_FOR_SECOND_CONV_LAYER (NUMBER_OF_WINDOWS_POOLINGLAYER1 - (KERNEL_SIZE - 1))
// E.g.: if the length of the pooled sequences (input seqs. to convolution) is 54, then
// the length of the convolution is 54 - 5 = 49.
#define NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER 12
#define NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER1

// LEVEL 2 -> POOLING
#define SEQUENCE_LENGTH_INPUT_POOLINGLAYER2 SEQUENCE_LENGTH_FOR_SECOND_CONV_LAYER
#define NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER2 NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER
#define NUMBER_OF_WINDOWS_POOLINGLAYER2 (SEQUENCE_LENGTH_INPUT_POOLINGLAYER2 / WINDOW_LENGTH_FOR_POOLING)

// LEVEL 3 -> CONVOLUTION
#define SEQUENCE_LENGTH_INPUT_THIRD_CONV_LAYER NUMBER_OF_WINDOWS_POOLINGLAYER2
#define SEQUENCE_LENGTH_FOR_THIRD_CONV_LAYER (NUMBER_OF_WINDOWS_POOLINGLAYER2 - (KERNEL_SIZE - 1))
// E.g.: if the length of the pooled sequences (input seqs. to convolution) is 24, then
// the length of the convolution is 24 - 5 = 19.
#define NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER 18
#define NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER2

// LEVEL 3 -> POOLING
#define SEQUENCE_LENGTH_INPUT_POOLINGLAYER3 SEQUENCE_LENGTH_FOR_THIRD_CONV_LAYER
#define NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER3 NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER
#define NUMBER_OF_WINDOWS_POOLINGLAYER3 (SEQUENCE_LENGTH_INPUT_POOLINGLAYER3 / WINDOW_LENGTH_FOR_POOLING)

// LEVEL 4 -> CONVOLUTION
#define SEQUENCE_LENGTH_INPUT_FOURTH_CONV_LAYER NUMBER_OF_WINDOWS_POOLINGLAYER3
#define SEQUENCE_LENGTH_FOR_FOURTH_CONV_LAYER (NUMBER_OF_WINDOWS_POOLINGLAYER3 - (KERNEL_SIZE - 1))
// E.g.: if the length of the pooled sequences (input seqs. to convolution) is 9, then
// the length of the convolution is 9 - 5 = 4.
#define NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER 24
#define NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER3

// LEVEL 4 -> POOLING
#define SEQUENCE_LENGTH_INPUT_POOLINGLAYER4 SEQUENCE_LENGTH_FOR_FOURTH_CONV_LAYER
#define NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER4 NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER
#define NUMBER_OF_WINDOWS_POOLINGLAYER4 (SEQUENCE_LENGTH_INPUT_POOLINGLAYER4 / WINDOW_LENGTH_FOR_POOLING)

// FLATTEN
#define SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER (NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER4 * NUMBER_OF_WINDOWS_POOLINGLAYER4)

// DENSE (OUTPUT LAYER)
#define NUMBER_OF_CLASSES 3

// DESPLAY INTERMEDIATE/INTERNAL TENSORS IN THE CNN
// #define DEBUG_INTERMEDIATE // If defined then the intermediate tensors are shown.
//                            // If not defined, then only the input and output tensors will be shown

// FUNCTION DEFINITIONS




bool LOAD_MODEL_PARAMETERS( const char *PATH_TO_MODEL_PARAMETERS, 
                            float (*weights_for_first_conv_layer)[KERNEL_SIZE],
                            float (*weights_for_second_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER][KERNEL_SIZE], 
                            float (*weights_for_third_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER][KERNEL_SIZE], 
                            float (*weights_for_fourth_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER][KERNEL_SIZE],
                            float (*weights_for_dense_layer)[SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER], 
                            float (*biases_for_first_conv_layer),
                            float (*biases_for_second_conv_layer),
                            float (*biases_for_third_conv_layer), 
                            float (*biases_for_fourth_conv_layer), 
                            float (*biases_for_dense_layer)
                            );



bool inference_for_prediction(
                            float (*input_sequence_to_first_conv),
                            float (*output_network_sequence), 
                            float (*weights_for_first_conv_layer)[KERNEL_SIZE],
                            float (*weights_for_second_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER][KERNEL_SIZE], 
                            float (*weights_for_third_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER][KERNEL_SIZE], 
                            float (*weights_for_fourth_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER][KERNEL_SIZE],
                            float (*weights_for_dense_layer)[SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER], 
                            float (*biases_for_first_conv_layer),
                            float (*biases_for_second_conv_layer),
                            float (*biases_for_third_conv_layer), 
                            float (*biases_for_fourth_conv_layer), 
                            float (*biases_for_dense_layer));

bool LEVEL1_convolution_for_single_sequence(float (*input_sequence), float (*output_sequence)[SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER],
                                            float (*weights_for_first_conv_layer)[KERNEL_SIZE],
                                            float(*biases_for_first_conv_layer));

bool LEVEL1_pooling_1D_sequences(float (*input_sequence)[SEQUENCE_LENGTH_INPUT_POOLINGLAYER1],
                                 float (*output_sequence)[NUMBER_OF_WINDOWS_POOLINGLAYER1]);

bool LEVEL2_convolution_for_multi_sequence(float (*input_sequence)[SEQUENCE_LENGTH_INPUT_SECOND_CONV_LAYER],
                                           float (*output_sequence)[SEQUENCE_LENGTH_FOR_SECOND_CONV_LAYER],
                                           float (*weights_for_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER][KERNEL_SIZE],
                                           float(*biases_for_conv_layer));

bool LEVEL2_pooling_1D_sequences(float (*input_sequence)[SEQUENCE_LENGTH_INPUT_POOLINGLAYER2],
                                 float (*output_sequence)[NUMBER_OF_WINDOWS_POOLINGLAYER2]);

bool LEVEL3_convolution_for_multi_sequence(
    float (*input_sequence)[SEQUENCE_LENGTH_INPUT_THIRD_CONV_LAYER],
    float (*output_sequence)[SEQUENCE_LENGTH_FOR_THIRD_CONV_LAYER],
    float (*weights_for_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER][KERNEL_SIZE],
    float(*biases_for_conv_layer));

bool LEVEL3_pooling_1D_sequences(float (*input_sequence)[SEQUENCE_LENGTH_INPUT_POOLINGLAYER3],
                                 float (*output_sequence)[NUMBER_OF_WINDOWS_POOLINGLAYER3]);

bool LEVEL4_convolution_for_multi_sequence(
    float (*input_sequence)[SEQUENCE_LENGTH_INPUT_FOURTH_CONV_LAYER],
    float (*output_sequence)[SEQUENCE_LENGTH_FOR_FOURTH_CONV_LAYER],
    float (*weights_for_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER][KERNEL_SIZE],
    float(*biases_for_conv_layer));

bool LEVEL4_pooling_1D_sequences(float (*input_sequence)[SEQUENCE_LENGTH_INPUT_POOLINGLAYER4],
                                 float (*output_sequence)[NUMBER_OF_WINDOWS_POOLINGLAYER4]);

bool flatten(float (*input_sequence)[NUMBER_OF_WINDOWS_POOLINGLAYER4], float(*output_sequence));

bool dense(float(*input_sequence), float(*output_sequence),
           float (*weights_for_dense_layer)[SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER],
           float(*biases_for_dense_layer));




// ###################################################
//  [IMPORTANT - STACK AND DATA SEGMENT WITHIN MEMORY]: 
//          The following variables are declared globally so that memory space is reserved in the data segment and not in the stack pile. 
//          The stack pile often has significantly less memory available, and thus attempting to reserve a large scope of memory locally (stack)
//          would likely lead to stack overflow with subsequent segmentation fault (memory violation)
//          NOTE: They are not stored in the heap since we are not using dynamic memory allocation


// Reserve memory for input data
uint16_t LINE_BUFFER[D_SAMPLES][D_CHANNELS];
    // We reserve memory for a buffer where we will read at once all the slit pixels within one line. In this manner, if we hav
    // 956 lines, then it means that we will have 956 reads from the disk. 
    // If we were to read only one single pixel at a time, the number of reads to the disk would be extremely high: 956 * 684 = 653 904 reads
    // Once one line buffer is filled with data, we will then pass only the individual pixels loaded for inference
    // NB: If we define the array as D_CHANNELS x D_SAMPLES, this would be less efficient since there would be a row jump per channel 
    //     For memory locality, since we need to process the channels together during inference, we define it instead as:
    //     D_SAMPLES x D_CHANNELS

    
// Reserve memory for output predictions
float PROBABILY_CONFIDENCE_MAP[D_LINES][D_SAMPLES];
//      To store the largest predicted probability for each pixel
uint8_t SEGMENTED_IMAGE[D_LINES][D_SAMPLES];
//     Each entry is either 0 (sea), 1 (land), or 2 (clouds)


// ###################################################


int main(int argc, char **argv)
{

    // #################################################################
    //          ADJUST THE PATHS 

    // Path to binarized model (in this case, in the same directory as the source code)
    const char *PATH_TO_MODEL_PARAMETERS="DEEP_LEARNING_NN_classifier_20240328_005008.bin";

    // Path to the .bip image (raw)
    const char *PATH_TO_IMAGE=
        "hsi0/binned_cube.bip";
        // "C:/Users/JONAJUSTO/Desktop/IMAGES_FOR_TESTING_INORBIT_DEPLOYMENT/39-20221213_CaptureDL_qatar_2022_12_13T06_34_13/qatar_2022-12-13.bip";
        // "C:/Users/JONAJUSTO/Desktop/IMAGES_FOR_TESTING_INORBIT_DEPLOYMENT/58-20221205_CaptureDL_blanca_2022_12_04T13_32_36/blanca_2022-12-04.bip";
        // "C:/Users/JONAJUSTO/Desktop/IMAGES_FOR_TESTING_INORBIT_DEPLOYMENT/59-20221205_CaptureDL_vigo_2022_12_04T11_35_18/vigo_2022-12-04.bip";
        // "C:/Users/JONAJUSTO/Desktop/IMAGES_FOR_TESTING_INORBIT_DEPLOYMENT/150-20220926_CaptureDL_00_kuwait_2022_09_26T06_58_45/kuwait_2022_09_26T06_58_45.bip";

        // "C:/Users/JONAJUSTO/Desktop/IMAGES_FOR_TESTING_INORBIT_DEPLOYMENT/207-20220808_CaptureDL_00_chao_2022_08_08T02_26_46/chao_2022_08_08T02_26_46.bip";

    // Segmented image output file
    const char *PATH_TO_SEGMENTED_IMAGE="hsi0/SEGMENTED_IMAGE.bin";

    if (argc == 1)
    {
        cout << "Usage:\n";
        cout << argv[0] << " <model parameters> [data cube] [output path]\n";
        cout << "\n";
        cout << "This program takes up to three arguments. These are in order:\n";
        cout << "1. Path to model parameters\n";
        cout << "2. Path to hyperspectral data cube (Default: " << PATH_TO_IMAGE << ")\n";
        cout << "3. Path to output segmented image (Default: " << PATH_TO_SEGMENTED_IMAGE << ")\n";
        return 1;
    }
    else
    {
        PATH_TO_MODEL_PARAMETERS = argv[1];
    }

    if (argc > 2)
    {
        PATH_TO_IMAGE = argv[2];
    }
    if (argc > 3)
    {
        PATH_TO_SEGMENTED_IMAGE = argv[3];
    }




    // #################################################################









    // #################################################################
    //          RESERVE MEMORY FOR THE MODEL'S PARAMETERS
    //  Consideration regarding spatial memory locality: 
    //      Although the compiler could make its own optimizations for the memory layout, 
    //      usually the variables will be laid in memory in the same order they are declared.
    //      Therefore, we declare them in the same order as the model's parameters are stored in
    //      the binary file to ensure access to contiguous memory

    float weights_for_first_conv_layer[NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER][KERNEL_SIZE];
    float weights_for_second_conv_layer[NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER]
                                       [NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER]
                                       [KERNEL_SIZE];
    float weights_for_third_conv_layer[NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER]
                                      [NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER]
                                      [KERNEL_SIZE];
    float weights_for_fourth_conv_layer[NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER]
                                       [NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER]
                                       [KERNEL_SIZE];
    float weights_for_dense_layer[NUMBER_OF_CLASSES][SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER];

    float biases_for_first_conv_layer[NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER]; 
    float biases_for_second_conv_layer[NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER];
    float biases_for_third_conv_layer[NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER];
    float biases_for_fourth_conv_layer[NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER];
    float biases_for_dense_layer[NUMBER_OF_CLASSES];

    // #################################################################



    LOAD_MODEL_PARAMETERS(PATH_TO_MODEL_PARAMETERS, 
                          weights_for_first_conv_layer, 
                          weights_for_second_conv_layer, 
                          weights_for_third_conv_layer, 
                          weights_for_fourth_conv_layer, 
                          weights_for_dense_layer, 
                          biases_for_first_conv_layer,
                          biases_for_second_conv_layer,
                          biases_for_third_conv_layer,
                          biases_for_fourth_conv_layer, 
                          biases_for_dense_layer);

    // cout << biases_for_dense_layer[0] << " " << biases_for_dense_layer[1] << " " << biases_for_dense_layer[2]<< endl; 
        // DEBUG: We can check the biases which are the last samples read in the binary file to crosscheck with python if
        //        the last samples in the binary file have been read ok





    cout << "Loading capture: " << PATH_TO_IMAGE << endl; 
    FILE *input_stream_to_capture=fopen(PATH_TO_IMAGE, "rb");

    if (input_stream_to_capture==NULL){
        cout << "ERROR WHEN OPENING FILE" << endl; 
        return 1; 
    } else{
        cout << "Capture opened ok! Starting to read it..." << endl; 
    }




    uint32_t number_of_samples_read_for_line_buffer; 
    for(uint16_t iterator_lines_in_bip=0; iterator_lines_in_bip<D_LINES; iterator_lines_in_bip++){
            // FILL IN ONE LINE BUFFER, i.e., all the samples within one line will be read and loaded into the buffer


            number_of_samples_read_for_line_buffer=
                        fread( LINE_BUFFER, BYTE_DEPTH_PER_SAMPLE_IN_CUBE, LINE_BUFFER_SIZE_IN_NUMBER_OF_SAMPLES, input_stream_to_capture);
                // (address, byte depth, number of samples to be read, input stream to file)
                // In the .bip capture, we have the following order: 
                //  S0CH0 S0CH1 ... S0CH119, S1CH0 S1CH1 ... S1CH119, ........, S683CH0 S683CH1 ... S683CH119
                // The LINE_BUFFER has dimensions (NB: row-major order in contiguous memory):
                //       LINE_BUFFER[D_SAMPLES][D_CHANNELS]
                // When writing into LINE_BUFFER, fread will write in the contiguous memories by increasing 2 bytes (uint16)
                // With the declaration LINE_BUFFER[D_SAMPLES][D_CHANNELS], the data in contiguous memory should be as follows: 
                // S0CH0 S0CH1 ... S0CH119, S1CH0 S1CH1 ... S1CH119, ........, S683CH0 S683CH1 ... S683CH119 <-- NB: Row-major order!
                // Therefore, in the reading we are using the correct order placing the channels and samples where they are supposed to be
                // In addition, we also account for spatial memory locality, since the channels within one pixel are stored in contiguous memory

            // Crosscheck if the correct size was read
            if (number_of_samples_read_for_line_buffer != LINE_BUFFER_SIZE_IN_NUMBER_OF_SAMPLES){
                cout << "Error when loading the capture! The line buffer was expected to have" << LINE_BUFFER_SIZE_IN_NUMBER_OF_SAMPLES 
                    << " values in total, however only" << number_of_samples_read_for_line_buffer << " values were read. The number of" 
                    << " values do not match! Exitting..." << endl;
                fclose(input_stream_to_capture);
                return 1;
            }
            // The line buffer was read ok

#pragma omp parallel for num_threads(2)
            for(uint16_t iterator_samples_in_line_buffer=0; iterator_samples_in_line_buffer<D_SAMPLES; iterator_samples_in_line_buffer++){
                // Execute the code next for each new sample to process within a line
                uint16_t PIXEL[D_CHANNELS];
                for(uint8_t iterator_channels_in_line_buffer=0; iterator_channels_in_line_buffer<D_CHANNELS; iterator_channels_in_line_buffer++){
                    // Note: LINE_BUFFER[D_SAMPLES][D_CHANNELS]
                    PIXEL[iterator_channels_in_line_buffer]=
                        LINE_BUFFER[iterator_samples_in_line_buffer][iterator_channels_in_line_buffer];                
                }
                // Pixel copied 


                // DEBUG: Uncomment the code below to show the values of a certain pixel and
                //        crosscheck with Python that the read is being done ok
                // // In Python: 
                // //      print(data_loader_o.DATA[IMAGE_NUMBER, LINES, SAMPLE, :])

                // if(iterator_lines_in_bip==955 && iterator_samples_in_line_buffer==683){
                //     cout << "DEBUG - Showing pixel values next: " << endl; 
                //     for(uint8_t iterator_for_channels=0; iterator_for_channels<D_CHANNELS; iterator_for_channels++){
                //         cout << PIXEL[iterator_for_channels] << " "; 
                //     }
                //     cout << endl; 
                // }



                // PRE-PROCESSING: REMOVE CHANNELS 
                std::vector<uint16_t> PIXEL_IN_VECTOR_FORMAT(PIXEL, PIXEL + D_CHANNELS);
                    // 1st argument: array to convert into vector
                    // 2nd argument: pointer to the first element after the end of the array

                std::vector<uint8_t> CHANNELS_TO_REMOVE=CHANNELS_TO_BE_REMOVED_IN_RAW_DATA;
                
                

                uint8_t number_of_channels_removed_SHIFT=0;
                 // Counter for the number of channels being removed
                uint8_t index_to_remove; 
                for (uint8_t channel : CHANNELS_TO_REMOVE) {
                    index_to_remove=channel-number_of_channels_removed_SHIFT;
                    // Assume these channels to remove: {10, 11, 12, 13, 116, 117, 118, 119}
                    // First iteration, the channel to remove will be in index: 10 - 0 = 10
                    // Second iteration, the channel to remove would be 11 but the indexes have shifted 1 position to the left due to the 1 channel removed
                    //               So, 11 - 1 = 10 
                    // Third iteration, the channel to remove would be 12 but the indexes have shifted 2 position to th left with respect to the original due to
                    //      the 2 channels removed. 
                    //              So, 12 -2 = 10

                    PIXEL_IN_VECTOR_FORMAT.erase(PIXEL_IN_VECTOR_FORMAT.begin() + index_to_remove);
                        // This is the advtange of using vectors in C++; it is way easier than what it would be in C
                        // As an argument to the function .erase() we pass the index to be removed in the vector

                    number_of_channels_removed_SHIFT++;
                }

                // Convert vector into array again
                uint16_t PIXEL_PROCESSED[D_CHANNELS_REDUCED];
                for(uint8_t iterator_reduced_channels=0; iterator_reduced_channels<D_CHANNELS_REDUCED; iterator_reduced_channels++){
                    PIXEL_PROCESSED[iterator_reduced_channels]=PIXEL_IN_VECTOR_FORMAT[iterator_reduced_channels];
                    // DEBUG: Uncomment the following code to crosscheck if the channels have been removed ok
                    //        It might be necessary to show the pixel before removing the channels to crosscheck 
                    // if(iterator_lines_in_bip==955 && iterator_samples_in_line_buffer==683)
                    //     cout <<  PIXEL_PROCESSED[iterator_reduced_channels] << " "; 
                }


                // PRE-PROCESSING: MIN-MAX NORMALIZATION

                uint16_t minimum_in_pixel, maximum_in_pixel; 
                minimum_in_pixel=maximum_in_pixel=PIXEL_PROCESSED[0];

                // At this point, we assume that in PIXEL_PROCESSED[0] we have the minimum value
                // At this ppint, we also assume that in PIXEL_PROCESSED[0] we have the maximum value
                for (uint16_t iterator_across_channels=1; iterator_across_channels<D_CHANNELS_REDUCED; iterator_across_channels++) {

                    // Update minimum next if needed
                    if(PIXEL_PROCESSED[iterator_across_channels] < minimum_in_pixel)
                        // If true, then it means that we have to update the minimum 
                        minimum_in_pixel=PIXEL_PROCESSED[iterator_across_channels];

                    // Update maximum next if needed
                    if(PIXEL_PROCESSED[iterator_across_channels] > maximum_in_pixel)
                        // If true, then it means that we have to update the maximum 
                        maximum_in_pixel=PIXEL_PROCESSED[iterator_across_channels];
    
                }

                // Apply normalization
                uint16_t range=maximum_in_pixel - minimum_in_pixel; 
                    // Precalculating the normalization factor in the denominator
                float PIXEL_PROCESSED_NORMALIZED[D_CHANNELS_REDUCED];
                    // IMPORTANT: The values after the normalization become float

                for(uint8_t iterator_across_channels=0; iterator_across_channels<D_CHANNELS_REDUCED; iterator_across_channels++){
                    PIXEL_PROCESSED_NORMALIZED[iterator_across_channels]=
                        static_cast<float>(PIXEL_PROCESSED[iterator_across_channels] - minimum_in_pixel) / range; 
                            // IMPORTANT: Make the casting to float, otherwise uint16 will be either 0 or 1

                    // DEBUG: Uncomment the following code to crosscheck with Python if the normalized values are ok
                    // if(iterator_lines_in_bip==100 && iterator_samples_in_line_buffer==200)
                    //     cout << PIXEL_PROCESSED_NORMALIZED[iterator_across_channels] << " "; 
                }

                // DEBUG: Uncomment the following code to crosscheck that the minimum, maximum, and range are calculated ok
                // if(iterator_lines_in_bip==100 && iterator_samples_in_line_buffer==200)
                //     cout << " *** " << minimum_in_pixel << " " << maximum_in_pixel << " " << range << endl; 
            





                // ######################################################
                //          INFERENCE

                // Reserve memory for the probability vector that will be resulting from the inference
                float output_network_sequence[NUMBER_OF_CLASSES];

                // RUN INFERENCE 
                // Inference is run over PIXEL_PROCESSED_NORMALIZED, result is given in output_network_sequence
                inference_for_prediction(PIXEL_PROCESSED_NORMALIZED,
                                         output_network_sequence, 
                                         weights_for_first_conv_layer,
                                         weights_for_second_conv_layer,
                                         weights_for_third_conv_layer,
                                         weights_for_fourth_conv_layer,
                                         weights_for_dense_layer,
                                         biases_for_first_conv_layer,
                                         biases_for_second_conv_layer,
                                         biases_for_third_conv_layer, 
                                         biases_for_fourth_conv_layer, 
                                         biases_for_dense_layer);
                // The output_network_sequence consists of a probability vector - e.g.: 3 classes, then 3 probabilities

                // ######################################################

                // Check which is the highest probability and for which class
                float highest_probabily=output_network_sequence[0];
                uint8_t index_of_class_with_highest_probability=0; 
                    // For 3 classes, the possible values are: 0, 1, or 2.
                for(uint8_t iterator_for_output_probabilities=1; iterator_for_output_probabilities<NUMBER_OF_CLASSES; iterator_for_output_probabilities++){
                    if(output_network_sequence[iterator_for_output_probabilities] > highest_probabily){
                        // Update highest probability and its corresponding class index
                        highest_probabily=output_network_sequence[iterator_for_output_probabilities]; 
                        index_of_class_with_highest_probability=iterator_for_output_probabilities;    
                    }
                }

                // // DEBUG: The following is only for debug
                // cout << highest_probabily << " " << static_cast<int> (index_of_class_with_highest_probability); 
                //     // The static_cast<int> is needed just because of the std::cout



                // The results to keep are: 
                //              highest_probabily (float) ---> to store in float PROBABILY_CONFIDENCE_MAP[D_LINES][D_SAMPLES]
                //              index_of_class_with_highest_probability (uint8_t) --> to store in uint8_t SEGMENTED_IMAGE[D_LINES][D_SAMPLES]

                PROBABILY_CONFIDENCE_MAP[iterator_lines_in_bip][iterator_samples_in_line_buffer]=highest_probabily; 
                SEGMENTED_IMAGE[iterator_lines_in_bip][iterator_samples_in_line_buffer]=index_of_class_with_highest_probability; 

                // Now move to next pixel within the line buffer
            }
            // All pixels within the line buffer are processed
            // Ready to move to the next line

            cout << "Inference run ok for line number: " << iterator_lines_in_bip << endl; 
                
            
    }

    // Processing is completed

    // No need to free up memory since dynamic memory allocation is not being used, with prospect of future FPGA implementation

    fclose(input_stream_to_capture);
        // The data is processed, we close the stream to the capture






    // AT THIS POINT, WE HAVE THE CAPTURE SEGMENTED

    FILE *file = fopen(PATH_TO_SEGMENTED_IMAGE, "wb"); 
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    fwrite(SEGMENTED_IMAGE, sizeof(uint8_t), D_LINES * D_SAMPLES, file);
    fclose(file);


    // for(uint16_t index_for_samples=0; index_for_samples<D_SAMPLES; index_for_samples++){
    //     cout << static_cast<int>(SEGMENTED_IMAGE[250][index_for_samples]) << " ";

    // }


    cout << "REACHES THE END OK"; 
    return 10; 





    // cout << endl
    //      << "Output sequence DENSE (OUTPUT LAYER): " << endl;
    // for (uint8_t i = 0; i < NUMBER_OF_CLASSES; i++)
    // {
    //     cout << output_network_sequence[i] << " ";
    // }
    // return 0;


}




bool LOAD_MODEL_PARAMETERS( const char *PATH_TO_MODEL_PARAMETERS, 
                            float (*weights_for_first_conv_layer)[KERNEL_SIZE],
                            float (*weights_for_second_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER][KERNEL_SIZE], 
                            float (*weights_for_third_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER][KERNEL_SIZE], 
                            float (*weights_for_fourth_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER][KERNEL_SIZE],
                            float (*weights_for_dense_layer)[SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER], 
                            float (*biases_for_first_conv_layer),
                            float (*biases_for_second_conv_layer),
                            float (*biases_for_third_conv_layer), 
                            float (*biases_for_fourth_conv_layer), 
                            float (*biases_for_dense_layer)
                            ){

        
    cout << "Loading model: " << PATH_TO_MODEL_PARAMETERS << endl; 
    FILE *input_stream_to_file=fopen(PATH_TO_MODEL_PARAMETERS, "rb");


    if (input_stream_to_file==NULL){
        cout << "ERROR WHEN OPENING FILE" << endl; 
        return 1; 
    } else{
        cout << "File opened ok! Starting to read it..." << endl; 
    }

    // LOAD WEIGHTS FOR LEVEL 1 : CONVOLUTION
    //  To be loaded into:     
    //      float weights_for_first_conv_layer[NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER][KERNEL_SIZE];

    uint8_t number_of_samples_read; 
    for(uint8_t r_neurons=0; r_neurons<NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER; r_neurons++){
        for(uint8_t c_kernel_size=0; c_kernel_size<KERNEL_SIZE; c_kernel_size++){
            // READ ONE SAMPLE AND STORE IT
            number_of_samples_read=
                fread(&(weights_for_first_conv_layer[r_neurons][c_kernel_size]),
                    BYTE_DEPTH_OF_FLOAT, 1, input_stream_to_file);

            // CROSSCHECK ONE SAMPLE WAS READ
            if (number_of_samples_read !=1){
                perror("Error when loading the input file!!");
                fclose(input_stream_to_file);
                return 1;
            }

                // About input arguments in fread: 
                    // 1st: address
                    // 2nd: bytes to be read
                    // 3rd: number of elements
                    // 4th: input stream being read
                    // Note: EOF should not occur since the number of samples to be read is controlled by the loop
        }
    }



    // LOAD WEIGHTS FOR LEVEL 2 : CONVOLUTION
    //  To be loaded into:    
    /*      float weights_for_second_conv_layer[NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER]
                                       [NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER]
                                       [KERNEL_SIZE];
    */ 
   // NB: Each weight will now have several components

    for(uint8_t p_neurons=0; p_neurons<NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER; p_neurons++){
        for(uint8_t r_components=0; r_components<NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER; r_components++){
            for(uint8_t c_kernel_size=0; c_kernel_size<KERNEL_SIZE; c_kernel_size++){
                // READ ONE SAMPLE AND STORE IT
                number_of_samples_read=
                    fread(&(weights_for_second_conv_layer[p_neurons][r_components][c_kernel_size]),
                        BYTE_DEPTH_OF_FLOAT, 1, input_stream_to_file);

                // CROSSCHECK ONE SAMPLE WAS READ
                if (number_of_samples_read !=1){
                    perror("Error when loading the input file!!");
                    fclose(input_stream_to_file);
                    return 1;
                }
            }
        }
    }


    // LOAD WEIGHTS FOR LEVEL 3 : CONVOLUTION
    //  To be loaded into:    
    /*      float weights_for_third_conv_layer[NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER]
                                       [NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER]
                                       [KERNEL_SIZE];
    */ 
   // NB: Each weight will now have several components

    for(uint8_t p_neurons=0; p_neurons<NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER; p_neurons++){
        for(uint8_t r_components=0; r_components<NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER; r_components++){
            for(uint8_t c_kernel_size=0; c_kernel_size<KERNEL_SIZE; c_kernel_size++){
                // READ ONE SAMPLE AND STORE IT
                number_of_samples_read=
                    fread(&(weights_for_third_conv_layer[p_neurons][r_components][c_kernel_size]),
                        BYTE_DEPTH_OF_FLOAT, 1, input_stream_to_file);

                // CROSSCHECK ONE SAMPLE WAS READ
                if (number_of_samples_read !=1){
                    perror("Error when loading the input file!!");
                    fclose(input_stream_to_file);
                    return 1;
                }
            }
        }
    }




    // LOAD WEIGHTS FOR LEVEL 4 : CONVOLUTION
    //  To be loaded into:    
    /*      float weights_for_fourth_conv_layer[NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER]
                                       [NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER]
                                       [KERNEL_SIZE];
    */ 
   // NB: Each weight will now have several components

    for(uint8_t p_neurons=0; p_neurons<NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER; p_neurons++){
        for(uint8_t r_components=0; r_components<NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER; r_components++){
            for(uint8_t c_kernel_size=0; c_kernel_size<KERNEL_SIZE; c_kernel_size++){
                // READ ONE SAMPLE AND STORE IT
                number_of_samples_read=
                    fread(&(weights_for_fourth_conv_layer[p_neurons][r_components][c_kernel_size]),
                        BYTE_DEPTH_OF_FLOAT, 1, input_stream_to_file);

                // CROSSCHECK ONE SAMPLE WAS READ
                if (number_of_samples_read !=1){
                    perror("Error when loading the input file!!");
                    fclose(input_stream_to_file);
                    return 1;
                }
            }
        }
    }





    // LOAD WEIGHTS FOR DENSE (OUTPUT LAYER)
    //  To be loaded into:     
    //          float weights_for_dense_layer[NUMBER_OF_CLASSES][SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER];


    for(uint8_t r_class_neurons=0; r_class_neurons<NUMBER_OF_CLASSES; r_class_neurons++){
        for(uint8_t c_connetions_for_input_features=0; 
            c_connetions_for_input_features<SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER;
                 c_connetions_for_input_features++){
            // READ ONE SAMPLE AND STORE IT
            number_of_samples_read=
                fread(&(weights_for_dense_layer[r_class_neurons][c_connetions_for_input_features]),
                    BYTE_DEPTH_OF_FLOAT, 1, input_stream_to_file);
            // CROSSCHECK ONE SAMPLE WAS READ
            if (number_of_samples_read !=1){
                perror("Error when loading the input file!!");
                fclose(input_stream_to_file);
                return 1;
            }
        }
    }



    // LOAD BIASES FOR LEVEL 1 : CONVOLUTION
    //  To be loaded into:     
    //      float biases_for_first_conv_layer[NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER]; 

    for(uint8_t iterator_neurons=0; iterator_neurons<NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER; iterator_neurons++){
            // READ ONE SAMPLE AND STORE IT
            number_of_samples_read=
                fread(&(biases_for_first_conv_layer[iterator_neurons]),
                    BYTE_DEPTH_OF_FLOAT, 1, input_stream_to_file);
            // CROSSCHECK ONE SAMPLE WAS READ
            if (number_of_samples_read !=1){
                perror("Error when loading the input file!!");
                fclose(input_stream_to_file);
                return 1;
            }

    }


    // LOAD BIASES FOR LEVEL 2 : CONVOLUTION
    //  To be loaded into: 
    //      float biases_for_second_conv_layer[NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER];

    for(uint8_t iterator_neurons=0; iterator_neurons<NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER; iterator_neurons++){
            // READ ONE SAMPLE AND STORE IT
            number_of_samples_read=
                fread(&(biases_for_second_conv_layer[iterator_neurons]),
                    BYTE_DEPTH_OF_FLOAT, 1, input_stream_to_file);
            // CROSSCHECK ONE SAMPLE WAS READ
            if (number_of_samples_read !=1){
                perror("Error when loading the input file!!");
                fclose(input_stream_to_file);
                return 1;
            }

    }


    // LOAD BIASES FOR LEVEL 3 : CONVOLUTION
    //  To be loaded into: 
    //      float biases_for_third_conv_layer[NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER];


    for(uint8_t iterator_neurons=0; iterator_neurons<NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER; iterator_neurons++){
            // READ ONE SAMPLE AND STORE IT
            number_of_samples_read=
                fread(&(biases_for_third_conv_layer[iterator_neurons]),
                    BYTE_DEPTH_OF_FLOAT, 1, input_stream_to_file);
            // CROSSCHECK ONE SAMPLE WAS READ
            if (number_of_samples_read !=1){
                perror("Error when loading the input file!!");
                fclose(input_stream_to_file);
                return 1;
            }

    }


    // LOAD BIASES FOR LEVEL 4 : CONVOLUTION
    //  To be loaded into: 
    //      float biases_for_fourth_conv_layer[NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER];

    for(uint8_t iterator_neurons=0; iterator_neurons<NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER; iterator_neurons++){
            // READ ONE SAMPLE AND STORE IT
            number_of_samples_read=
                fread(&(biases_for_fourth_conv_layer[iterator_neurons]),
                    BYTE_DEPTH_OF_FLOAT, 1, input_stream_to_file);
            // CROSSCHECK ONE SAMPLE WAS READ
            if (number_of_samples_read !=1){
                perror("Error when loading the input file!!");
                fclose(input_stream_to_file);
                return 1;
            }

    }


    // LOAD BIASES FOR DENSE (OUTPUT LAYER)
    //  To be loaded into: 
    //      float biases_for_dense_layer[NUMBER_OF_CLASSES];


    for(uint8_t iterator_neurons=0; iterator_neurons<NUMBER_OF_CLASSES; iterator_neurons++){
            // READ ONE SAMPLE AND STORE IT
            number_of_samples_read=
                fread(&(biases_for_dense_layer[iterator_neurons]),
                    BYTE_DEPTH_OF_FLOAT, 1, input_stream_to_file);
            // CROSSCHECK ONE SAMPLE WAS READ
            if (number_of_samples_read !=1){
                perror("Error when loading the input file!!");
                fclose(input_stream_to_file);
                return 1;
            }

    }


    fclose(input_stream_to_file);

    return true; 
}







bool inference_for_prediction(
                            float (*input_sequence_to_first_conv),
                            float (*output_network_sequence), 
                            float (*weights_for_first_conv_layer)[KERNEL_SIZE],
                            float (*weights_for_second_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER][KERNEL_SIZE], 
                            float (*weights_for_third_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER][KERNEL_SIZE], 
                            float (*weights_for_fourth_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER][KERNEL_SIZE],
                            float (*weights_for_dense_layer)[SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER], 
                            float (*biases_for_first_conv_layer),
                            float (*biases_for_second_conv_layer),
                            float (*biases_for_third_conv_layer), 
                            float (*biases_for_fourth_conv_layer), 
                            float (*biases_for_dense_layer)){

    // Model parameters:
        // float biases_for_first_conv_layer[NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER]; 
        // float weights_for_first_conv_layer[NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER][KERNEL_SIZE];

    float output_sequence_from_fist_conv[NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER][SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER];
    // The convolution result for each kernel over the input sequence gives as a result a 1D convolution sequence
    // Since there are various kernels, the result will be in a 2D tensor
    // NB: the output is float32 (same as the default data type in Keras)

    // #######################
    // LEVEL 1: 1D CONVOLUTION

    LEVEL1_convolution_for_single_sequence(input_sequence_to_first_conv, output_sequence_from_fist_conv,
                                           weights_for_first_conv_layer, biases_for_first_conv_layer);

    // The output has dimensions:
    //      NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER x SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER

    // ########################

#ifdef DEBUG_INTERMEDIATE
    cout << endl
         << "Output sequences LEVEL 1 (CONVOLUTION): " << endl;
    for (uint8_t n = 0; n < NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER; n++)
    {
        cout << "Sequence for kernel number " << to_string(n) << ": " << endl;
        for (uint8_t l = 0; l < SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER; l++)
        {
            cout << output_sequence_from_fist_conv[n][l] << " ";
        }
        cout << endl;
    }
#endif

    // At this point, output_sequence_from_fist_conv has dimensions:
    //      NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER x SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER

    float output_sequence_from_first_pool[NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER1][NUMBER_OF_WINDOWS_POOLINGLAYER1];
    // Length of the output sequence: matches the number of windows since there will be one sample produced per window

    // #######################
    // LEVEL 1: POOLING

    LEVEL1_pooling_1D_sequences(output_sequence_from_fist_conv, output_sequence_from_first_pool);
    // The output has dimensions:
    //      NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER1 x NUMBER_OF_WINDOWS_POOLINGLAYER1

    // #######################

#ifdef DEBUG_INTERMEDIATE
    cout << endl
         << "Output sequences LEVEL 1 (POOLING): " << endl;
    for (uint8_t n = 0; n < NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER1; n++)
    {
        cout << "Sequence number " << to_string(n) << ": " << endl;
        for (uint8_t m = 0; m < NUMBER_OF_WINDOWS_POOLINGLAYER1; m++)
        {
            cout << output_sequence_from_first_pool[n][m] << " ";
        }
        cout << endl;
    }
#endif

                             
    //// Model parameters:
    //      float biases_for_second_conv_layer[NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER];
    //      float weights_for_second_conv_layer[NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER]
    //                                    [NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER]
    //                                    [KERNEL_SIZE];


    float output_sequence_from_second_conv_layer
        [NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER][SEQUENCE_LENGTH_FOR_SECOND_CONV_LAYER];

    // #######################
    // LEVEL 2: CONVOLUTION

    LEVEL2_convolution_for_multi_sequence(output_sequence_from_first_pool,
                                          output_sequence_from_second_conv_layer,
                                          weights_for_second_conv_layer,
                                          biases_for_second_conv_layer);

    // The output has dimensions:
    //      NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER x SEQUENCE_LENGTH_FOR_SECOND_CONV_LAYER

    // #######################

#ifdef DEBUG_INTERMEDIATE
    cout << endl
         << "Output sequences LEVEL 2 (CONVOLUTION): " << endl;

    for (uint8_t n = 0; n < NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER; n++)
    {
        cout << "Sequence number " << to_string(n) << ": " << endl;
        for (uint8_t l = 0; l < SEQUENCE_LENGTH_FOR_SECOND_CONV_LAYER; l++)
        {
            cout << output_sequence_from_second_conv_layer[n][l] << " ";
        }
        cout << endl;
    }
#endif

    // At this point, output_sequence_from_second_conv_layer has dimensions:
    //      NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER x SEQUENCE_LENGTH_FOR_SECOND_CONV_LAYER

    float output_sequence_from_second_pool[NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER2][NUMBER_OF_WINDOWS_POOLINGLAYER2];
    // Length of the output sequence: matches the number of windows since there will be one sample produced per window

    // #######################
    // LEVEL 2: POOLING

    LEVEL2_pooling_1D_sequences(output_sequence_from_second_conv_layer, output_sequence_from_second_pool);
    // The output has dimensions:
    //      NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER2 x NUMBER_OF_WINDOWS_POOLINGLAYER2

    // #######################

#ifdef DEBUG_INTERMEDIATE
    cout << endl
         << "Output sequences LEVEL 2 (POOLING): " << endl;
    for (uint8_t n = 0; n < NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER2; n++)
    {
        cout << "Sequence number " << to_string(n) << ": " << endl;
        for (uint8_t m = 0; m < NUMBER_OF_WINDOWS_POOLINGLAYER2; m++)
        {
            cout << output_sequence_from_second_pool[n][m] << " ";
        }
        cout << endl;
    }
#endif

    // At this point, output_sequence_from_second_pool has dimensions:
    //       NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER2 x NUMBER_OF_WINDOWS_POOLINGLAYER2


    // Model parameters:
        // float biases_for_third_conv_layer[NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER];
        // float weights_for_third_conv_layer[NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER]
        //                                 [NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER]
        //                                 [KERNEL_SIZE];


    float output_sequence_from_third_conv_layer
        [NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER][SEQUENCE_LENGTH_FOR_THIRD_CONV_LAYER];

    // #######################
    // LEVEL 3: CONVOLUTION

    LEVEL3_convolution_for_multi_sequence(output_sequence_from_second_pool,
                                          output_sequence_from_third_conv_layer,
                                          weights_for_third_conv_layer,
                                          biases_for_third_conv_layer);

    // The output has dimensions:
    //       NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER x SEQUENCE_LENGTH_FOR_THIRD_CONV_LAYER

    // #######################

#ifdef DEBUG_INTERMEDIATE
    cout << endl
         << "Output sequences LEVEL 3 (CONVOLUTION): " << endl;
    for (uint8_t n = 0; n < NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER; n++)
    {
        cout << "Sequence number " << to_string(n) << ": " << endl;
        for (uint8_t l = 0; l < SEQUENCE_LENGTH_FOR_THIRD_CONV_LAYER; l++)
        {
            cout << output_sequence_from_third_conv_layer[n][l] << " ";
        }
        cout << endl;
    }
#endif

    // At this point, output_sequence_from_third_conv_layer has dimensions:
    //       NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER x SEQUENCE_LENGTH_FOR_THIRD_CONV_LAYER

    float output_sequence_from_third_pool[NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER3][NUMBER_OF_WINDOWS_POOLINGLAYER3];
    // Length of the output sequence: matches the number of windows since there will be one sample produced per window

    // #######################
    // LEVEL 3: POOLING

    LEVEL3_pooling_1D_sequences(output_sequence_from_third_conv_layer, output_sequence_from_third_pool);

    // The output has dimensions:
    //       NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER3 x NUMBER_OF_WINDOWS_POOLINGLAYER3

    // #######################

#ifdef DEBUG_INTERMEDIATE
    cout << endl
         << "Output sequences LEVEL 3 (POOLING): " << endl;
    for (uint8_t n = 0; n < NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER3; n++)
    {
        cout << "Sequence number " << to_string(n) << ": " << endl;
        for (uint8_t m = 0; m < NUMBER_OF_WINDOWS_POOLINGLAYER3; m++)
        {
            cout << output_sequence_from_third_pool[n][m] << " ";
        }
        cout << endl;
    }
#endif

    // At this point, output_sequence_from_third_pool has dimensions:
    //      NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER3 x NUMBER_OF_WINDOWS_POOLINGLAYER3


    // Model parameters:
        // float biases_for_fourth_conv_layer[NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER];
        // float weights_for_fourth_conv_layer[NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER]
        //                                 [NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER]
        //                                 [KERNEL_SIZE];



    float output_sequence_from_fourth_conv_layer
        [NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER][SEQUENCE_LENGTH_FOR_FOURTH_CONV_LAYER];

    // #######################
    // LEVEL 4: CONVOLUTION

    LEVEL4_convolution_for_multi_sequence(output_sequence_from_third_pool,
                                          output_sequence_from_fourth_conv_layer,
                                          weights_for_fourth_conv_layer,
                                          biases_for_fourth_conv_layer);

    // The output has dimensions:
    //       NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER x SEQUENCE_LENGTH_FOR_FOURTH_CONV_LAYER

    // #######################

#ifdef DEBUG_INTERMEDIATE
    cout << endl
         << "Output sequences LEVEL 4 (CONVOLUTION): " << endl;
    for (uint8_t n = 0; n < NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER; n++)
    {
        cout << "Sequence number " << to_string(n) << ": " << endl;
        for (uint8_t l = 0; l < SEQUENCE_LENGTH_FOR_FOURTH_CONV_LAYER; l++)
        {
            cout << output_sequence_from_fourth_conv_layer[n][l] << " ";
        }
        cout << endl;
    }
#endif

    // At this point, output_sequence_from_fourth_conv_layer has dimensions:
    //       NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER x SEQUENCE_LENGTH_FOR_FOURTH_CONV_LAYER

    float output_sequence_from_fourth_pool[NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER4][NUMBER_OF_WINDOWS_POOLINGLAYER4];
    // Length of the output sequence: matches the number of windows since there will be one sample produced per window

    // #######################
    // LEVEL 4: POOLING

    LEVEL4_pooling_1D_sequences(output_sequence_from_fourth_conv_layer, output_sequence_from_fourth_pool);

    // The output has dimensions:
    //       NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER4 x NUMBER_OF_WINDOWS_POOLINGLAYER4

    // #######################

#ifdef DEBUG_INTERMEDIATE
    cout << endl
         << "Output sequences LEVEL 4 (POOLING): " << endl;
    for (uint8_t n = 0; n < NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER4; n++)
    {
        cout << "Sequence number " << to_string(n) << ": " << endl;
        for (uint8_t m = 0; m < NUMBER_OF_WINDOWS_POOLINGLAYER4; m++)
        {
            cout << output_sequence_from_fourth_pool[n][m] << " ";
        }
        cout << endl;
    }
#endif

    // At this point, output_sequence_from_fourth_pool has dimensions:
    //      NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER4 x NUMBER_OF_WINDOWS_POOLINGLAYER4

    float output_sequence_flatten[SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER];

    // #######################
    // FLATTEN

    flatten(output_sequence_from_fourth_pool, output_sequence_flatten);

    // The output has dimensions (1D):
    //      SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER

    // #######################

#ifdef DEBUG_INTERMEDIATE
    cout << endl
         << "Output sequence FLATTEN: " << endl;
    for (uint8_t i = 0; i < SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER; i++)
    {
        cout << output_sequence_flatten[i] << " ";
    }
#endif

    // At this point, output_sequence_flatten has dimension (1D):
    //      SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER


    // Model parameters:
        // float biases_for_dense_layer[NUMBER_OF_CLASSES];
        // float weights_for_dense_layer[NUMBER_OF_CLASSES][SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER];


    // float output_network_sequence[NUMBER_OF_CLASSES];
    //      In this case, this is passes as an argument to the function


    // #######################
    // DENSE (OUTPUT LAYER)

    dense(output_sequence_flatten, output_network_sequence, weights_for_dense_layer, biases_for_dense_layer);

    // The output has dimensions (1D):
    //       NUMBER_OF_CLASSES

    // #######################

    return true; 

}








bool LEVEL1_pooling_1D_sequences(float (*input_sequence)[SEQUENCE_LENGTH_INPUT_POOLINGLAYER1],
                                 float (*output_sequence)[NUMBER_OF_WINDOWS_POOLINGLAYER1])
{
    // If the input_sequence was a 1D array, then you could declare the input argument as float *input_sequence since there's only one row
    // However, a 2D array is considered an "array of arrays (sequences)". In the syntax, we need to specify the compiler the "limit/boundary" for each
    // row since all the memory storage within each rows and across all the rows is contiguous and hence it is needed to determine the memory position where a
    // a new row starts

    for (uint8_t s = 0; s < NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER1; s++)
    {
        // s is the iterator for the sequences
        for (uint8_t i = 0; i < NUMBER_OF_WINDOWS_POOLINGLAYER1; i++)
        { // Iterate the entries within a single sequence
            // i is iterator for the windows, the first element to each window is given by
            //  i*WINDOW_LENGTH_FOR_POOLING, e.g., if window is 2, then 0*2=0, then 1*2=2, then 2*2=4, etc
            //  so it gives the proper first index at the start of each window
            // If it was the case that we have for instance a window size of 2, and a length of e.g. 7, then we avoid removing
            // the last sample to save computation time - it is not needed to be removed since there will be only 3 windows and
            // when the three windows with the 6 samples are processed, then it's already done and the for-loop won't try to iterate
            // into an incomplete 4th window
            uint8_t j = i * WINDOW_LENGTH_FOR_POOLING;
            output_sequence[s][i] = (input_sequence[s][j] >= input_sequence[s][j + 1]) ? input_sequence[s][j] : input_sequence[s][j + 1];
        }
        // At this point, we move to the next sequence
    }

    return true;
}

bool LEVEL2_pooling_1D_sequences(float (*input_sequence)[SEQUENCE_LENGTH_INPUT_POOLINGLAYER2],
                                 float (*output_sequence)[NUMBER_OF_WINDOWS_POOLINGLAYER2])
{
    for (uint8_t s = 0; s < NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER2; s++)
    {
        for (uint8_t i = 0; i < NUMBER_OF_WINDOWS_POOLINGLAYER2; i++)
        {
            uint8_t j = i * WINDOW_LENGTH_FOR_POOLING;
            output_sequence[s][i] = (input_sequence[s][j] >= input_sequence[s][j + 1]) ? input_sequence[s][j] : input_sequence[s][j + 1];
        }
    }
    return true;
}

bool LEVEL3_pooling_1D_sequences(float (*input_sequence)[SEQUENCE_LENGTH_INPUT_POOLINGLAYER3],
                                 float (*output_sequence)[NUMBER_OF_WINDOWS_POOLINGLAYER3])
{
    for (uint8_t s = 0; s < NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER3; s++)
    {
        for (uint8_t i = 0; i < NUMBER_OF_WINDOWS_POOLINGLAYER3; i++)
        {
            uint8_t j = i * WINDOW_LENGTH_FOR_POOLING;
            output_sequence[s][i] = (input_sequence[s][j] >= input_sequence[s][j + 1]) ? input_sequence[s][j] : input_sequence[s][j + 1];
        }
    }
    return true;
}

bool LEVEL4_pooling_1D_sequences(float (*input_sequence)[SEQUENCE_LENGTH_INPUT_POOLINGLAYER4],
                                 float (*output_sequence)[NUMBER_OF_WINDOWS_POOLINGLAYER4])
{
    for (uint8_t s = 0; s < NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER4; s++)
    {
        for (uint8_t i = 0; i < NUMBER_OF_WINDOWS_POOLINGLAYER4; i++)
        {
            uint8_t j = i * WINDOW_LENGTH_FOR_POOLING;
            output_sequence[s][i] = (input_sequence[s][j] >= input_sequence[s][j + 1]) ? input_sequence[s][j] : input_sequence[s][j + 1];
        }
    }
    return true;
}

bool LEVEL1_convolution_for_single_sequence(float (*input_sequence), float (*output_sequence)[SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER],
                                            float (*weights_for_first_conv_layer)[KERNEL_SIZE],
                                            float(*biases_for_first_conv_layer))
{

    float sliding_input_window[KERNEL_SIZE];
    for (uint8_t n = 0; n < NUMBER_OF_NEURAL_KERNELS_FIRST_CONV_LAYER; n++)
    {
        // This loop iterates across the neural kernels 0, 1, 2, 3, etc.
        // The process next will be repeated for each individual neural kernel

        for (uint8_t i = 0; i < KERNEL_SIZE; i++)
        {
            // Initialize the sliding window
            sliding_input_window[i] = input_sequence[i];

            // 1st kernel: sliding_input_window only has reserbved memory, so we write the 1st samples from the input_sequence
            // 2nd and rest of kernels: sliding_input_window has the last samples from the input sequence and hence we "restart" it again
            // cout << sliding_input_window[i] << endl;  // Only for debug
        }

        for (uint8_t j = 0; j < SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER; j++)
        {
            // Comments for this for-loop:
            //      Each iteration results in a sample from the convolution
            //      The operations apply to the sliding window at this stage (same MAC computations as a dense layer, but
            //      applied to the sliding window)
            float MAC_accumulator = 0;
            // Input samples are uint16_t, but model parameters are float32, hence the result in accumulator will be float32
            // In Keras, the output of the layers is also float32 by default

            // Debug: To crosscheck the values of the sliding window in each iteration uncomment the following code
            // cout << "debug next  :";
            // for(uint8_t i=0; i<KERNEL_SIZE; i++){
            //     cout << sliding_input_window[i] << " ";
            // }
            // cout << endl;

            for (uint8_t i = 0; i < KERNEL_SIZE; i++)
            {
                // MAC loop
                MAC_accumulator += sliding_input_window[i] * weights_for_first_conv_layer[n][i];
            }
            MAC_accumulator += biases_for_first_conv_layer[n];
            // Add bias
            MAC_accumulator = (MAC_accumulator > 0) ? MAC_accumulator : 0;
            // ReLU activation

            // At this point, MAC_accumulator already has one sample to be written into the output_sequence (convolution result)

            output_sequence[n][j] = MAC_accumulator;
            // Write the resulting sample

            // Slide next the window for the next iteration
            if (j == (SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER - 1))
            { // NOTHING TO SLIDE

                // At this point, WE HAVE JUST WRITTTEN THE LAST SAMPLE OF THE CONVOLUTION RESULT
                // No need to move the sliding window since attempting to move it further to the right would result in
                // memory violation/trash

                // Example: SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER = 109 -> the samples from 0 to 108 have been processed at this
                //          point and the last one has been the 108, e.g.: 108 109 110 111 112 113. If we move the sliding window, we would be
                //          attempting to shift an out-of-bound memory (114/trash).

                ; // Do nothing
                // When this point is reached, then the foor loop will increase j++ and it j will be j=SEQUENCE_LENGTH_FOR_FIRST_CONV_LAYER.
                // Therefore, this a condition mismatch and the loop for producing the convolution samples is broken
                // We are DONE WITH THE RESULT FOR ONE NEURON
                // We now move to the next neural kernel (if there are still kernels left).
            }
            else
            { // SOMETHING TO SLIDE

                for (uint8_t i = 0; i < (KERNEL_SIZE - 1); i++)
                {
                    // Shift the window to the right
                    sliding_input_window[i] = sliding_input_window[i + 1];
                    // Important KERNEL_SIZE-1 to avoid memory segment violation (or trash) in the following scenario
                    // Filter of kernel size = 6 -> sliding input window indexes from 0 to 5.
                    // If we have i<KERNEL_SIZE, then when trying to access the last position in the window, this will occur:
                    // sliding_input_window[5]=sliding_input_window[6] (memory violation/trash!)
                }
                sliding_input_window[KERNEL_SIZE - 1] = input_sequence[KERNEL_SIZE + j];
                // We access the last index in the sliding window to write a new sample in
                // This new element cannot be taken from the sliding_input_window as it would be out of bound, and
                // also the element we need is not in sliding_input_window but in input_sequence
                // Example: input sequence=[0, 1, 2, 3, 4, 5, 6, 7, ...]
                // For the first time we slide the window -> j=0, we incorporate element 6 (6 + 0)
                // For the second time we slide the window -> j=1, we incorporate element 7 (6 + 1)
                // For the third time we slide the window -> j=2, we incorporate element 8 (6 + 2)
                /*
                    First window: [0 1 2  3 4 5].
                    Second window: [1 2 3  4 5 6]
                    Third window: [2 3 4  5 6 7]
                */
            }
        }
    }
    return true;
}

bool LEVEL2_convolution_for_multi_sequence(
    float (*input_sequence)[SEQUENCE_LENGTH_INPUT_SECOND_CONV_LAYER],
    float (*output_sequence)[SEQUENCE_LENGTH_FOR_SECOND_CONV_LAYER],
    float (*weights_for_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER][KERNEL_SIZE],
    float(*biases_for_conv_layer))
{

    float sliding_input_window[NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER][KERNEL_SIZE];

    for (uint8_t n = 0; n < NUMBER_OF_NEURAL_KERNELS_SECOND_CONV_LAYER; n++)
    {
        // Repeat the operations below for each individual neuron

        for (uint8_t r = 0; r < NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER; r++)
        {
            // Repeat the operations below for each input component
            // Note: the number of input components is given by the number of neurons in the previous convolutional layer.
            //       However, we avoid using the notation "n" to avoid confusion with the number of neurons of the current convolutional layer.

            for (uint8_t c = 0; c < KERNEL_SIZE; c++)
            {
                // For a component in the input, fill in the window (initialisation)
                sliding_input_window[r][c] = input_sequence[r][c];
                // The idea is similar to the window in the convolution for single sequence, but now the only difference is that
                // since the convolution is applied to multisequence, then the input window is in 2D since the input data is also in 2D
            }
        }

        // MAC operations below
        for (uint8_t j = 0; j < SEQUENCE_LENGTH_FOR_SECOND_CONV_LAYER; j++)
        {
            // Each iteration produces a sample resulting from the convolution operation for a certain neuron

            float MAC_accumulator = 0;

            for (uint8_t r = 0; r < NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER; r++)
            {
                // With r, we iterate the components/sequences of the sliding window and the filter
                for (uint8_t c = 0; c < KERNEL_SIZE; c++)
                {
                    // With c, we interate the samples within each component/sequence of the sliding window and the filter
                    MAC_accumulator += sliding_input_window[r][c] * weights_for_conv_layer[n][r][c];
                }
            }
            // At this point, we have iterated the complete 2D window, and we have applied the corresponding MACs

            MAC_accumulator += biases_for_conv_layer[n];
            MAC_accumulator = (MAC_accumulator > 0) ? MAC_accumulator : 0;

            // Resulting sample from convolution ready to be written

            output_sequence[n][j] = MAC_accumulator;

            // Sliding the input window next
            // NOTE: for more comments, check the single sequence function - the idea is the same, the difference is that now
            //       we work with several sequences rather than just with one sequence
            if (j == (SEQUENCE_LENGTH_FOR_SECOND_CONV_LAYER - 1))
            {
                ; // Do nothing
            }
            else
            {
                for (uint8_t s = 0; s < NUMBER_OF_INPUT_SEQUENCES_FOR_SECOND_CONV_LAYER; s++)
                {
                    // This is to slide the window across each sequence/component
                    for (uint8_t c = 0; c < (KERNEL_SIZE - 1); c++)
                    {

                        sliding_input_window[s][c] = sliding_input_window[s][c + 1];
                    }
                    sliding_input_window[s][KERNEL_SIZE - 1] = input_sequence[s][KERNEL_SIZE + j];
                }
            }
        }
    }
    return true;
}

bool LEVEL3_convolution_for_multi_sequence(
    float (*input_sequence)[SEQUENCE_LENGTH_INPUT_THIRD_CONV_LAYER],
    float (*output_sequence)[SEQUENCE_LENGTH_FOR_THIRD_CONV_LAYER],
    float (*weights_for_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER][KERNEL_SIZE],
    float(*biases_for_conv_layer))
{

    float sliding_input_window[NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER][KERNEL_SIZE];

    for (uint8_t n = 0; n < NUMBER_OF_NEURAL_KERNELS_THIRD_CONV_LAYER; n++)
    {
        for (uint8_t r = 0; r < NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER; r++)
        {
            for (uint8_t c = 0; c < KERNEL_SIZE; c++)
            {
                sliding_input_window[r][c] = input_sequence[r][c];
            }
        }

        for (uint8_t j = 0; j < SEQUENCE_LENGTH_FOR_THIRD_CONV_LAYER; j++)
        {
            float MAC_accumulator = 0;
            for (uint8_t r = 0; r < NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER; r++)
            {
                for (uint8_t c = 0; c < KERNEL_SIZE; c++)
                {
                    MAC_accumulator += sliding_input_window[r][c] * weights_for_conv_layer[n][r][c];
                }
            }
            MAC_accumulator += biases_for_conv_layer[n];
            MAC_accumulator = (MAC_accumulator > 0) ? MAC_accumulator : 0;
            output_sequence[n][j] = MAC_accumulator;

            if (j == (SEQUENCE_LENGTH_FOR_THIRD_CONV_LAYER - 1))
            {
                ; // Do nothing
            }
            else
            {
                for (uint8_t s = 0; s < NUMBER_OF_INPUT_SEQUENCES_FOR_THIRD_CONV_LAYER; s++)
                {
                    for (uint8_t c = 0; c < (KERNEL_SIZE - 1); c++)
                    {

                        sliding_input_window[s][c] = sliding_input_window[s][c + 1];
                    }
                    sliding_input_window[s][KERNEL_SIZE - 1] = input_sequence[s][KERNEL_SIZE + j];
                }
            }
        }
    }
    return true;
}

bool LEVEL4_convolution_for_multi_sequence(
    float (*input_sequence)[SEQUENCE_LENGTH_INPUT_FOURTH_CONV_LAYER],
    float (*output_sequence)[SEQUENCE_LENGTH_FOR_FOURTH_CONV_LAYER],
    float (*weights_for_conv_layer)[NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER][KERNEL_SIZE],
    float(*biases_for_conv_layer))
{

    float sliding_input_window[NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER][KERNEL_SIZE];

    for (uint8_t n = 0; n < NUMBER_OF_NEURAL_KERNELS_FOURTH_CONV_LAYER; n++)
    {
        for (uint8_t r = 0; r < NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER; r++)
        {
            for (uint8_t c = 0; c < KERNEL_SIZE; c++)
            {
                sliding_input_window[r][c] = input_sequence[r][c];
            }
        }

        for (uint8_t j = 0; j < SEQUENCE_LENGTH_FOR_FOURTH_CONV_LAYER; j++)
        {
            float MAC_accumulator = 0;
            for (uint8_t r = 0; r < NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER; r++)
            {
                for (uint8_t c = 0; c < KERNEL_SIZE; c++)
                {
                    MAC_accumulator += sliding_input_window[r][c] * weights_for_conv_layer[n][r][c];
                }
            }
            MAC_accumulator += biases_for_conv_layer[n];
            MAC_accumulator = (MAC_accumulator > 0) ? MAC_accumulator : 0;
            output_sequence[n][j] = MAC_accumulator;

            if (j == (SEQUENCE_LENGTH_FOR_FOURTH_CONV_LAYER - 1))
            {
                ; // Do nothing
            }
            else
            {
                for (uint8_t s = 0; s < NUMBER_OF_INPUT_SEQUENCES_FOR_FOURTH_CONV_LAYER; s++)
                {
                    for (uint8_t c = 0; c < (KERNEL_SIZE - 1); c++)
                    {

                        sliding_input_window[s][c] = sliding_input_window[s][c + 1];
                    }
                    sliding_input_window[s][KERNEL_SIZE - 1] = input_sequence[s][KERNEL_SIZE + j];
                }
            }
        }
    }
    return true;
}

bool flatten(float (*input_sequence)[NUMBER_OF_WINDOWS_POOLINGLAYER4],
             float(*output_sequence))
{
    /*

    In Keras, the flatten layer works as follows:

    Matrix to flatten:
        Row 0 -> 1, 2
        Row B -> 3, 4
    Flatten matrix:
        Unique row: 1, 3, 2, 4
    NOTE: In Keras, the column is often understood as having in its rows the components
    */

    // In the most nested loop we do not iterate columns within a row (as it is regular the case),
    // but we iterate the rows within a column instead - NB: less efficient in terms of memory locality
    for (uint8_t i = 0, j = 0; i < NUMBER_OF_WINDOWS_POOLINGLAYER4; i++)
    {
        // j: Iterator for the entries in the output sequence in 1D (flattened)
        for (uint8_t s = 0; s < NUMBER_OF_INPUT_AND_OUTPUT_SEQUENCES_POOLINGLAYER4; s++)
        {
            output_sequence[j++] = input_sequence[s][i];
        }
    }
    return true;
}

bool dense(float(*input_sequence), float(*output_sequence),
           float (*weights_for_dense_layer)[SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER],
           float(*biases_for_dense_layer))
{

    // input_sequence: 1D as it is the result from the flatten layer
    // output_sequence: 1D as it is an array of probabilities
    // Note: 3 classes to segment -> 3 neurons
    //       If there are e.g. 4 inputs to the dense layer, each neuron will have 4 connections

    float logits_vector[NUMBER_OF_CLASSES];
    // Neurons output before activation with softmax to convert logits into probablities
    // The results from the MAC accumulator will be stored here

    for (uint8_t n = 0; n < NUMBER_OF_CLASSES; n++)
    {
        float MAC_accumulator = 0;
        // Initialise it here, not possible in the for-loop as MAC_accumulator is of different type as iterator
        for (uint8_t i = 0; i < SEQUENCE_LENGTH_OUTPUT_FLATTENLAYER; i++)
        {
            MAC_accumulator += input_sequence[i] * weights_for_dense_layer[n][i];
        }
        // At this point, MAC_accumulator has not accounted yet for the bias (pre-bias & pre-activation value)
        MAC_accumulator += biases_for_dense_layer[n];
        // (post-bias & pre-activation value: logit)
        // At this point, MAC_accumulator accounts for the bias. MAC_accumulator has a "logit"
        logits_vector[n] = MAC_accumulator;
    }

    // Softmax activation next
    float NORMALIZATION_FACTOR = 0;
    // We calculate the normalization (denominator in the softmax formula)
    for (uint8_t i = 0; i < NUMBER_OF_CLASSES; i++)
    {
        output_sequence[i] = exp(logits_vector[i]);
        // Exponential for each neuron (for the numerator in the softmax formula)
        // Note: exponential will later be normalized
        NORMALIZATION_FACTOR += output_sequence[i];
        // As we calculate the exponencial for each neuron, we sum them (for the denominator in the softmax formula)
    }

    for (uint8_t i = 0; i < NUMBER_OF_CLASSES; i++)
    {
        // Now that we have the final normalization factor calculated, we take the exponential of each neuron and
        // divide it by the normalization factor
        output_sequence[i] = output_sequence[i] / NORMALIZATION_FACTOR;
        // We overwrite the exponencials by the normalization factor
    }
    // At this point, output_sequence has the probabilities
    return true;
}
