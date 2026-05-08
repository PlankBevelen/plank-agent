module.exports = {
  apps: [
    {
      name: 'plank-agent',
      script: 'python',
      args: 'flask/app.py',
      exec_mode: 'fork',
      instances: 1,
      cwd: __dirname,
      out_file: 'logs/out.log',
      error_file: 'logs/error.log',
      merge_logs: true,
      autorestart: true,
      time: true
    }
  ]
};
