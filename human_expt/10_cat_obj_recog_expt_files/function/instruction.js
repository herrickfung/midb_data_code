// contains all instruction text & survey text

var parti_info_questions = [
  [
    {
      type: 'text',
      prompt: "GTID number: ",
      required: true,
    },
    {
      type: 'text',
      prompt: "Age: ",
      required: true,
    },
  ],

  [
    {
      type: 'drop-down',
      prompt: "Gender",
      options: ["Female", "Male", "Transgender", "Non-binary/Non-conforming", "Others", "Prefer not to say"],
      required: true,
    },
    {
      type: 'drop-down',
      prompt: "Ethnicity: ",
      options: ["Hispanic or Latino", "Not Hispanic or Latino", "Prefer not to say"],
      required: true,
    },
    {
      type: 'drop-down',
      prompt: "Race: ",
      options: ["American Indian / Alaska Native", "Asian", "Native Hawaiian or Other Pacific Islander", "Black or African American", "White", "More than one race", "Prefer not to say"],
      required: true,
    },
    {
      type: 'drop-down',
      prompt: "Vision: ",
      options: ["Normal", "Corrected-to-Normal (using glasses/contacts)", "Color-blind", "Others"],
      required: true,
    },
    {
      type: 'drop-down',
      prompt: "Handedness",
      options: ["Left-Handed", "Right-Handed", "Ambidextrous"],
      required: true,
    },
  ],
];

var welcome_text =[
  '<p style="font-size:20px;color:#000000;text-align:left">'+
  'Welcome to the experiment! Thank you for your participation.' +
  '<br> In this experiment, we are interested in how people make quick perceptual decisions.' +
  '<br><br> You may navigate back and forth with the buttons below or <br> with the Left and Right Arrow on your keyboard.' +
  '</p>',

  '<p style="font-size:20px;color:#000000;text-align:left">'+
   'We ask that you comply with the following experiment requirements:'+
  '<br><br> 1) In order for the experiment to run smoothly, stop any downloads or processes <br> that strain your internet connection.'+
  '<br><br> 2) Do not use your web browser\'s Back or Refresh buttons at any point during this experiment.'+
  '<br><br> 3) This experiment requires good concentration. As such, we ask that you complete the experiment <br> in an environment that is as free as possible of noise and distraction.'+
  '<br><br> Thank you for your cooperation.'+
  '</p>',

  '<p style="font-size:20px;color:#000000;text-align:left">'+
  'Most people complete this experiment in about 60 minutes. <br><br> You will complete 2 practice blocks and 4 runs of 4 experimental blocks, <br>with each block containing 50 trials.' +
  '<br><br> Each block will take about 4 minutes, and you can take break after each block.' +
  '<br><br> Please do your best to take the experiment in a single sitting <br> without excessive interruptions or taking very long breaks.'+
  '</p>',

  '<p style="font-size:20px;color:#000000;text-align:left">'+
  'This experiment is about object recognition.'+
  '<br><br>You will see an image, choose its category, and give confidence rating on your decision.' +
  '<br><br> More detailed task instruction will be given later.'+
  '<br><br> Now, we will switch to the fullscreen mode, <br>please DO NOT escape the fullscreen mode until the experiment is completed.'+
  '</p>',
];

var task_instruction = [
// On each trial, you will see quickly-presented orientation patches close to 45 degrees.
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'The following screens are the instruction of the task, please read carefully.' +
  '<br><br> On each trial, you will see a quickly-presented image.' +
  '<br><br> You may navigate the instruction back and forth with the buttons below or <br> with the Left and Right Arrow on your keyboard.' +
  '</p>',

  '<img src = "10_cat_obj_recog_expt_files/img/expt_stim/train/img_072_train_7_1.png" width = 350px text-align = center>' +
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'This is an example of the stimulus.' +
  '<br><br> In the experiment, you will see images and judge which 1 of the 10 categories does the stimulus belong to.' +
  '</p>',

  '<img src = "10_cat_obj_recog_expt_files/img/support_img/resp_screen_1.png" width = 800px text-align = center>' +
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'This is the list of 10 object categories that you will choose from.' +
  '<br><br>You do NOT have to memorize this list.'+
  '<br><br>This list will be presented on the screen until you choose a category.' +
  '</p>',

  '<img src = "10_cat_obj_recog_expt_files/img/support_img/resp_screen_2.png" width = 800px text-align = center>' +
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'You should have seen a cat on the second page.' +
  '<br><br>Thus, you should move the cursor to the cat box and click on it.'+
  '<br><br>You cannot change your decision after clicking onto the category.'+
  '</p>',

  '<img src = "10_cat_obj_recog_expt_files/img/support_img/conf_screen.png" width = 800px text-align = center>' +
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'You will then be asked to indicate your confidence on a scale from 0% to 100%.'+
  '<br><br> <b>0%</b> means that you believe you get the decision <b>wrong</b>'+
  '<br><br> <b>10%</b> means that you are <b>guessing</b>'+
  '<br><br> <b>100%</b> means that you are completely certain that you got it <b>right</b>.'+
  '<br><br> Move the cursor to the box that best described your level of confidence.'+
  '<br><br> Use the scale to reflect your confidence as faithfully as possible.'+
  '</p>',

  '<img src = "10_cat_obj_recog_expt_files/img/support_img/category.png" width = 800px text-align = center>' +
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'Now, Take some time to look at these categories and try to remember the location of each category. '+
  '<br> These location will remain the same in the experiment.'+
  '</p>',

  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'Great, you have completed the instructions.' +
  '<br><br> Please go back and read the instruction if you are unclear about what to do.' +
  '<br><br> Otherwise, press next and get ready for the practice block!' +
  '</p>',
  ];

var start_prac_instruct_1 =
    '<p style="font-size:20px;color:#000000;text-align:left">' +
    'Now, you will work on 10 practice trials with minimal noise <br> to familiarize with the category and the task.' +
    '<br><br> Make sure to <br><br>(1) Do your best, and <br>(2) Report confidence faithfully!' +
    "<br><br> If you're ready, click begin to start the practice." +
    '</p>';

var start_prac_instruct_2 =
    '<p style="font-size:20px;color:#000000;text-align:left">' +
    'Good Job!' +
    '<br><br> We used a very low noise level in the previous block. <br> In the actual experiment, you will see noisier images.' +
    '<br><br> In the following practice, you will complete 20 trials that <br> resemble what you will see in an actual experimental block.' +
    '<br><br> Always make sure to — <br><br>(1) Do your best and <br>(2) Report confidence faithfully!' +
    "<br><br> If you're ready, click begin to start the practice." +
    '</p>';


var start_experiment_instruct =
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'Cool! You\'ve completed the practice, take a short break.' +
  '<br><br> Now, you will work on 4 runs of 4 experimental blocks,' +
  '<br> with each experimental block containing 50 trials.' +
  '<br><br> Each block will take about 4 minutes and you can take a break after each block.' +
  '<br><br> No trial feedback will be given. Instead, you will know your score after each block.' +
  '<br><br> Try to increase your score by — <br><br>(1) Doing your best and <br>(2) Reporting confidence faithfully!' +
  '<br><br> Click the button below whenever you are ready for the experiment!' +
  '</p>';

var debrief_text =
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  "Excellent! You've completed the whole experiment." +
  "</p>" +
  '<p style="font-size:26px;color:#FF0000;text-align:left">' +
  "DATA UPLOADING, DON'T CLOSE THE WINDOW!" +
  "</p>" +
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  "You will be redirected once it is completed." +
  "<br><br> You will also be prompt to save a csv file once it is completed, <br> you are strongly advised to keep a copy of the file just in case any technical issue occured." +
  "<br> You may delete the file permanently after you receive approval from the study." +
  "<br> The researchers will contact you only if they need you to send the file manually." +
  "<br> Otherwise, just wait until the approval happen, which will usually be within 24 hours." +
  "<br><br> Thank you for your participation, have a good day, and I hope to see you again!" +
  '</p>';


