var scrapeAudio = function(patientId) {
  if (!patientId) {
    patientId = "PATIENT_DATA_RENAME_ME"
  }
  var oddRows = document.getElementsByClassName('odd');
  var evenRows = document.getElementsByClassName('even');

  if (oddRows.length !== 21 && evenRows.length !== 21) {
    console.log("INCORRECT NUMBER OF RECORDINGS");
    return;
  }

  var rows = [];
  for (var i = 0; i < 21; i++) {
    rows.push(oddRows[i]);
    rows.push(evenRows[i]);
  }

  rows.sort(function(row1, row2) {
    if (row1.id < row2.id) {
      return 1
    } else {
      return -1
    }
  })

  var recordings = [
    "TF/TF_RLL_1.wav",
    "TF/TF_LLL_1.wav",
    "TF/TF_RML_1.wav",
    "TF/TF_LML_1.wav",
    "TF/TF_RUL_1.wav",
    "TF/TF_LUL_1.wav",
    "BS/BS_RLL_3.wav",
    "BS/BS_LLL_3.wav",
    "BS/BS_RML_3.wav",
    "BS/BS_LML_3.wav",
    "BS/BS_RUL_3.wav",
    "BS/BS_LUL_3.wav",
    "BS/BS_RLL_2.wav",
    "BS/BS_LLL_2.wav",
    "BS/BS_RML_2.wav",
    "BS/BS_LML_2.wav",
    "BS/BS_RUL_2.wav",
    "BS/BS_LUL_2.wav",
    "BS/BS_RLL_1.wav",
    "BS/BS_LLL_1.wav",
    "BS/BS_RML_1.wav",
    "BS/BS_LML_1.wav",
    "BS/BS_RUL_1.wav",
    "BS/BS_LUL_1.wav",
    "PS/PS_RLL_3.wav",
    "PS/PS_LLL_3.wav",
    "PS/PS_RML_3.wav",
    "PS/PS_LML_3.wav",
    "PS/PS_RUL_3.wav",
    "PS/PS_LUL_3.wav",
    "PS/PS_RLL_2.wav",
    "PS/PS_LLL_2.wav",
    "PS/PS_RML_2.wav",
    "PS/PS_LML_2.wav",
    "PS/PS_RUL_2.wav",
    "PS/PS_LUL_2.wav",
    "PS/PS_RLL_1.wav",
    "PS/PS_LLL_1.wav",
    "PS/PS_RML_1.wav",
    "PS/PS_LML_1.wav",
    "PS/PS_RUL_1.wav",
    "PS/PS_LUL_1.wav",
  ];

  var urls = [];

  var urlToCommand = function(i) {
    return "sleep 2; curl -o " + recordings[i] + " \"" + urls[i] + "\";";
  }

  var generateScript = function() {
    if (urls.length !== recordings.length) {
      console.log("FAILURE! Incorrect number of urls found: " + urls.length);
      return
    }

    for (var i = 0; i < recordings.length; i++) {
      if (!urls[i]) {
        console.log("FAILURE! Missing url for file " + recordings[i]);
        console.log("index: " + i);
        return
      }
      for (var j = i + 1; j < recordings.length; j++) {
        if (urls[i] === urls[j]) {
          console.log("FAILURE! Duplicate urls found for " + recordings[i] + " and " + recordings[j]);
          console.log("index: " + i + " and " + j);
          return
        }
      }
    }

    function wrapCommand(cmd) {
      return "echo \"" + cmd.replace(/"/g, "\\\"") + "\" >> download.sh";    
    }
    var cmds = [
      wrapCommand("cd ~/Desktop; rm download.sh; mkdir " + patientId + "; cd " + patientId),
      wrapCommand("mkdir BS; mkdir PS; mkdir TF;")
    ];
    for (var i = 0; i < recordings.length; i++) {
      cmds.push(wrapCommand(urlToCommand(i)));
    }
    cmds.push(wrapCommand("open ../"))
    cmds.push("bash download.sh")
    var script = cmds.join("\n") + "\n";
    
    console.log("Success! Open your terminal, and copy and paste the entire script below:");
    console.log(script);
  }

  var scrapeAudioUrls = function(i) {
    var DELAY = 3000;

    if (i === recordings.length) {
      return generateScript();
    }
    var audioUrl = "https://dashboard.ekodevices.com/#/dashboard/recordings/" + rows[i].id;
    if (window.location.href !== audioUrl) {
      window.location.href = audioUrl;
      return setTimeout(function() { scrapeAudioUrls(i) }, DELAY)
    }
    var downloadElement = document.getElementById('download-sound-td')
    if (!downloadElement) {
      return setTimeout(function() { scrapeAudioUrls(i) }, DELAY)
    }
    if (!downloadElement.children[0].href) {
      return setTimeout(function() { scrapeAudioUrls(i) }, DELAY)
    }

    urls.push(downloadElement.children[0].href)

    scrapeAudioUrls(i + 1)
  }

  scrapeAudioUrls(0);
}
