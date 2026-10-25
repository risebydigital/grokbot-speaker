/* Milo showcase behaviour. No network calls.
   PLACEHOLDERS AWAITING ADAM - do not treat these as approved copy:
   1. Demo conversation part 2
   2. Demo conversation part 3
   3. Roadmap "next" line (the designer's holding text is kept)
*/
(function () {
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Part 1 is the approved designer copy. Parts 2 and 3 are placeholders. */
  var PARTS = [
    {
      you: "\u201cMilo, what are you?\u201d",
      milo: "I’m Milo, a home-made smart speaker that lives on your bookshelf. Say my name, ask me something, and I’ll talk it through with you."
    },
    {
      /* PLACEHOLDER awaiting Adam: demo conversation part 2. */
      you: "\u201cMilo, what can you do?\u201d",
      milo: "I listen for my name, then we talk. I answer out loud through the bookshelf speakers, and the Voice Ops Dashboard lets the team see how I’m getting on."
    },
    {
      /* PLACEHOLDER awaiting Adam: demo conversation part 3. */
      you: "\u201cMilo, how were you made?\u201d",
      milo: "In-house at Rise By Digital, from three parts that plug together: a single-board computer, a USB microphone and a pair of wired speakers."
    }
  ];

  /* credits is a real command. It is named in the section sentence and omitted
     from this help table so the panel matches the approved preview. */
  var COMMANDS = {
    help: "<span class=\"k\">Available commands</span>\n  <span class=\"c\">help</span>          <span class=\"d\">List commands</span>\n  <span class=\"c\">about</span>         <span class=\"d\">What Milo is</span>\n  <span class=\"c\">hardware</span>      <span class=\"d\">The parts list</span>\n  <span class=\"c\">architecture</span>  <span class=\"d\">How a conversation flows</span>\n  <span class=\"c\">install</span>       <span class=\"d\">Getting set up</span>\n  <span class=\"c\">roadmap</span>       <span class=\"d\">What’s next</span>\n",
    about: "<span class=\"k\">Milo</span>  <span class=\"d\">an RBD internal project</span>\nA home-made voice smart speaker. Say the wake word\n\"Milo\", then talk to it like a voice chat assistant.\nA Voice Ops Dashboard sits alongside the speaker.\n",
    hardware: "<span class=\"k\">Parts list</span>\n  [1] Single-board computer <span class=\"d\">(Pi 4)</span>\n  [2] USB microphone\n  [3] Wired bookshelf speakers <span class=\"d\">(a pair)</span>\n<span class=\"d\">Type 'architecture' to see how they talk to each other.</span>\n",
    architecture: "<span class=\"k\">How a conversation flows</span>\n  USB mic\n   \u2514\u2500\u25b6 wake word \"Milo\"\n        \u2514\u2500\u25b6 Voice AI\n             \u2514\u2500\u25b6 speakers\n  <span class=\"d\">Voice Ops Dashboard keeps watch</span>\n",
    install: "<span class=\"k\">Getting set up</span>\n<span class=\"d\">Private repo, RBD team access only.</span>\n  $ git clone \\\n    git@github.com:risebydigital/grokbot-speaker.git\n  $ cd grokbot-speaker\n<span class=\"d\">  # Full setup steps live in the README.</span>\n",
    /* PLACEHOLDER awaiting Adam: the "next" line is the designer's holding text. */
    roadmap: "<span class=\"k\">Roadmap</span>\n  now    Showcase site on GitHub Pages\n  next   <span class=\"d\">[items from the repo roadmap - to be confirmed]</span>\n",
    credits: "<span class=\"k\">Credits</span>\nThe voice loop is original code. These projects shaped it,\nand their sources were not copied:\n  m15-ai/Grok-Voice\n    <span class=\"d\">realtime voice, USB audio and PipeWire</span>\n  herrkaefer/snowman\n    <span class=\"d\">a fresh wake decoder for each listening period</span>\n  ErwanHeschung/Naka\n    <span class=\"d\">playback from a queue, plus a stalled-socket watchdog</span>\n"
  };

  var orb = document.getElementById("orb");
  var capYou = document.getElementById("cap-you");
  var capMilo = document.getElementById("cap-milo");
  var capCount = document.getElementById("cap-count");
  var capLive = document.getElementById("cap-live");
  var termLog = document.getElementById("term-log");
  var termForm = document.getElementById("term-form");
  var termInput = document.getElementById("term-input");
  var playToken = 0;
  var playing = false;
  var index = 0;
  var session = ["help", "hardware"];
  var typed = [];
  var typedAt = 0;

  function esc(value) {
    return value.replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function setMiloText(text) {
    var cursor = capMilo.querySelector(".cursor");
    Array.prototype.slice.call(capMilo.childNodes).forEach(function (node) {
      if (node !== cursor) capMilo.removeChild(node);
    });
    capMilo.insertBefore(document.createTextNode(text), cursor);
  }

  function setOrbState(talking, playingNow) {
    document.body.dataset.orb = talking ? "talking" : "idle";
    document.body.dataset.playing = playingNow ? "1" : "0";
    orb.setAttribute("aria-pressed", talking ? "true" : "false");
    orb.setAttribute(
      "aria-label",
      talking
        ? "Milo demo playing. Activate to stop."
        : "Play the Milo demo"
    );
  }

  function showPart(i, instant) {
    var part = PARTS[i];
    index = i;
    capYou.textContent = part.you;
    capCount.textContent = "Demo conversation \u00b7 " + (i + 1) + " of " + PARTS.length;
    if (instant || reduced) {
      setMiloText(part.milo);
      capLive.textContent = "You. " + part.you + " Milo. " + part.milo;
      return Promise.resolve();
    }
    setMiloText("");
    var token = playToken;
    return new Promise(function (resolve) {
      var n = 0;
      var step = function () {
        if (token !== playToken) {
          resolve();
          return;
        }
        n += 1;
        setMiloText(part.milo.slice(0, n));
        if (n >= part.milo.length) {
          capLive.textContent = "You. " + part.you + " Milo. " + part.milo;
          resolve();
          return;
        }
        window.setTimeout(step, 18);
      };
      step();
    });
  }

  function wait(ms) {
    return new Promise(function (resolve) {
      window.setTimeout(resolve, ms);
    });
  }

  function runFrom(start) {
    var token = ++playToken;
    playing = true;
    setOrbState(true, !reduced);
    var i = start;
    var pump = function () {
      if (token !== playToken) return;
      if (i >= PARTS.length) {
        playing = false;
        setOrbState(false, false);
        return;
      }
      showPart(i, false).then(function () {
        if (token !== playToken) return;
        return wait(reduced ? 2600 : 1800);
      }).then(function () {
        if (token !== playToken) return;
        i += 1;
        pump();
      });
    };
    pump();
  }

  function activate() {
    if (playing) {
      playToken += 1;
      playing = false;
      setOrbState(false, false);
      return;
    }
    runFrom(0);
  }

  function renderLog(cmds) {
    var prompt = '<span class="p">milo@bookshelf:~$</span> ';
    var html = "";
    cmds.forEach(function (cmd) {
      var body;
      if (Object.prototype.hasOwnProperty.call(COMMANDS, cmd)) {
        body = COMMANDS[cmd];
      } else {
        body = '<span class="d">command not found: ' + esc(cmd) + "\nType 'help' for the list.</span>\n";
      }
      html += prompt + '<span class="c">' + esc(cmd) + "</span>\n" + body + "\n";
    });
    termLog.innerHTML = html;
  }

  function setActiveChip(cmd) {
    document.querySelectorAll(".chip").forEach(function (chip) {
      var on = chip.getAttribute("data-cmd") === cmd;
      chip.classList.toggle("on", on);
      chip.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  function sizeInput() {
    var len = termInput.value.length;
    termInput.style.width = (len === 0 ? 0 : len) + "ch";
  }

  function runCommand(cmd) {
    session.push(cmd);
    typed.push(cmd);
    typedAt = typed.length;
    renderLog(session);
    setActiveChip(cmd);
    termLog.setAttribute("aria-live", "polite");
  }

  var params = new URLSearchParams(window.location.search);
  if (params.has("term")) {
    session = params.get("term").split(",").map(function (item) {
      return item.trim().toLowerCase();
    }).filter(Boolean);
  }
  renderLog(session);
  if (session.length) setActiveChip(session[session.length - 1]);

  var partParam = parseInt(params.get("part") || "1", 10);
  if (params.get("orb") === "talking") {
    var partIndex = Math.min(PARTS.length, Math.max(1, partParam)) - 1;
    playing = false;
    setOrbState(true, false);
    showPart(partIndex, true);
  }

  orb.addEventListener("click", activate);
  document.getElementById("hear").addEventListener("click", function () {
    document.getElementById("demo").scrollIntoView({ block: "nearest" });
    playToken += 1;
    playing = false;
    runFrom(0);
    orb.focus();
  });

  document.addEventListener("keydown", function (event) {
    if (event.key !== "Enter" && event.key !== " ") return;
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.target.closest("input, textarea, select, a, button, [contenteditable]")) return;
    event.preventDefault();
    activate();
  });

  termForm.addEventListener("submit", function (event) {
    event.preventDefault();
    var cmd = termInput.value.trim().toLowerCase();
    if (!cmd) return;
    termInput.value = "";
    sizeInput();
    runCommand(cmd);
  });

  termInput.addEventListener("input", sizeInput);
  termInput.addEventListener("keydown", function (event) {
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
    if (!typed.length) return;
    event.preventDefault();
    if (event.key === "ArrowUp") typedAt = Math.max(0, typedAt - 1);
    else typedAt = Math.min(typed.length, typedAt + 1);
    termInput.value = typedAt === typed.length ? "" : typed[typedAt];
    sizeInput();
  });

  document.querySelectorAll(".chip").forEach(function (chip) {
    chip.addEventListener("click", function () {
      runCommand(chip.getAttribute("data-cmd"));
      termInput.focus();
    });
  });

  sizeInput();
  document.documentElement.dataset.ready = "1";
})();
