// create triallist
var trials_to_break = 50;
triallist = [createPracticeTriallist1(),
  createPracticeTriallist2(),
  createTriallist()];
var break_trial_array = breakTriallist(triallist[2].length, trials_to_break);

// get subject ID from SONA;
var subject_id = getParameterByName("sona_id");

const jsPsych = initJsPsych({
  override_safe_mode: true,
  on_finish: function(){
    saveData("10ORE_" + subject_id + ".csv", jsPsych.data.get().csv());
    jsPsych.data.get().localSave('csv', "10ORE_" + subject_id + ".csv")
  },
});

////////////////////////////////////////////////////////////////////////////////
// main stimulus define for timeline

// preload stimulus trial-by-trial
var load = {
  type: jsPsychPreload,
  stimulus: getSrc,
  data: {
    curr_triallist: updateTriallist,
    trial_no: updateTrialNo,
    stimulus: 'preload',
  },
};

// main fixation stimulus_screen
var fixa = {
  type: jsPsychCanvasKeyboardResponse,
  stimulus: createFixation,
  choices: 'none',
  stimulus_duration: 300,
  trial_duration: 300,
  is_html: true,
  canvas_size: [400, 400],
  on_start: hideCursor,
  data: {
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,
    stimulus: 'fixation',
  },
};

var stim = {
  type: jsPsychCanvasKeyboardResponse,
  stimulus: createImages,
  stimulus_duration: 300,
  trial_duration: 300,
  is_html: true,
  choices: 'none',
  canvas_size: [400, 400],
  on_start: hideCursor,
  data: {
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,
    stimulus: 'stimulus',
  },
};

var resp = {
  type:jsPsychCanvasButtonResponse,
  stimulus: createBlank,
  choices: [
    'bed',
    'bridge',
    'car',
    'cat',
    'elephant',
    'flower',
    'house',
    'knife',
    'phone',
    'train'
  ],
  prompt: '<p style="position:absolute; left:100px; top:175px; margin:0; padding:0; font-size:24px; font-weight:bold">Select the object you saw below:</p>',
  canvas_size: [600, 600],
  button_html: createRespButtonConfig,
  on_start: showCursor,
  on_finish: checkAccuracy,
  data: {
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,
    image_index: 0,
    category: 0,
    blur: 0,
    reps: 0,
    correct: 0,
    stimulus: 'perceptual',
  },
};

var conf = {
  type:jsPsychCanvasButtonResponse,
  stimulus: createBlank,
  choices: ['0%<br>wrong', '10%<br>guess', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%<br>right'],
  prompt: '<p style="position:absolute; left:150px; top:225px; margin:0; padding:0; font-size:24px; font-weight:bold">How confident are you?</p>',
  canvas_size: [600, 600],
  button_html: createConfButtonConfig,
  on_start: showCursor,
  on_finish: getConfidence,
  data: {
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,
    running_score: 0,
    stimulus: 'confidence',
  },
};

var feed = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: createFeedbackScreen,
  choices: 'NO_KEYS',
  stimulus_duration: 1000,
  trial_duration: 1000,
  on_start: hideCursor,
  data: {
      datetime: getDatetime,
      subjectID: subject_id,
      curr_triallist: getCurrentTriallist,
      trial_no: getCurrentTrialNo,
      stimulus: "feedback",
    },
};


////////////////////////////////////////////////////////////////////////////////
// other minor variables in timeline

const browser_check = {
  type: jsPsychBrowserCheck,
  inclusion_function: (data) => {
    return ['chrome', 'firefox'].includes(data.browser) && data.mobile === false;
  },
  exclusion_message: (data) => {
    if (data.mobile === true){
      return '<p> You must use a desktop/laptop computer to participate in this experiment. </p>'
    } else if (!(['chrome', 'firefox'].includes(data.browser))){
      return '<p> You must use Chrome or Firefox as your browser to complete this experiment. </p>'
    }
  }
};

const getPartiInfo = {
  type: jsPsychSurvey,
  pages: parti_info_questions,
  title: "Please provide the following demographic information.",
};

const welcome = {
  type: jsPsychInstructions,
  pages: welcome_text,
  show_clickable_nav: true,
  show_page_number: true,
};

const chinrest = {
  type: jsPsychVirtualChinrest,
  blindspot_reps: 3,
  resize_units: "deg",
  pixels_per_unit: 30,
  item_path: "10_cat_obj_recog_expt_files/img/support_img/card.png",
  viewing_distance_report: 'none',
};

var practice_instructions = {
  type: jsPsychInstructions,
  pages: task_instruction,
  show_clickable_nav: true,
  show_page_number: true,
};

var start_practice = {
  type: jsPsychHtmlButtonResponse,
  stimulus: createStartPracticeScreen,
  choices: ['Begin Practice'],
  on_start: showCursor,
  data: {
      datetime: getDatetime,
      subjectID: subject_id,
      curr_triallist: getCurrentTriallist,
      trial_no: getCurrentTrialNo,
      stimulus: "start_practice"
    },
};

var start_experiment = {
  type: jsPsychHtmlButtonResponse,
  stimulus: start_experiment_instruct,
  choices: ['Begin Experiment'],
  on_start: showCursor,
  data: {
    datetime: getDatetime,
    subjectID: subject_id,
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,
    stimulus: "start experiment"
    },
};

var start_break_procedure = {
  type: jsPsychHtmlButtonResponse,
  stimulus: createStartBreakScreen,
  choices: ['Continue Experiment'],
  on_start: showCursor,
  data: {
      curr_triallist: getCurrentTriallist,
      trial_no: getCurrentTrialNo,
    },
};

var end_break_procedure = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: createEndBreakScreen,
  choices: " ",
  trial_duration: 1000,
  data: {
      curr_triallist: getCurrentTriallist,
      trial_no: getCurrentTrialNo,
    },
};

var debriefing = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: debrief_text,
  choices: "NO_KEYS",
  trial_duration: 60000,
  on_start: function(){
      saveData("data/16ORE_" + subject_id + ".csv", jsPsych.data.get().csv());
    },
};

var debriefing2 = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: debrief_text,
  choices: "NO_KEYS",
  trial_duration: 5000,
};

var startFullScreen = {
  type: jsPsychFullscreen,
  fullscreen_mode: true,
};

var endFullScreen = {
  type: jsPsychFullscreen,
  fullscreen_mode: false,
};

// end of define variables
////////////////////////////////////////////////////////////////////////////////
// the main experiment timeline starts here
var timeline = [];

timeline.push(browser_check);
timeline.push(getPartiInfo);
timeline.push(welcome);
timeline.push(startFullScreen);
timeline.push(chinrest);
timeline.push(practice_instructions);

for (let list = 0; list < triallist.length; list++){
  for (let trial = 0; trial < triallist[list].length; trial++){
    if (list === 2){
      // handle break trial
      if (break_trial_array.includes(trial) === true){
        timeline.push(start_break_procedure);
        timeline.push(end_break_procedure);
      };
      if (trial === 0){
        timeline.push(start_experiment);
      };
    } else {
      // for practice instructions
      if (trial === 0){
        timeline.push(start_practice);
      };
    };
    timeline.push(load)
    timeline.push(fixa);
    timeline.push(stim);
    timeline.push(resp);
    timeline.push(conf);
    // handle feedback for practice trials
    if (list != 2){
      timeline.push(feed);
    };
  };
};

timeline.push(debriefing2);
timeline.push(endFullScreen);
// // end of the whole experiment
// ////////////////////////////////////////////////////////////////////////////////

// // run experiment
jsPsych.run(timeline);
