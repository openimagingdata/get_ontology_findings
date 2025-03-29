personal task management:
long term:
    LLM finding editor as whole
    application takes a finding model
        json file corrsponfing to known schema
    app takes nat lang from user
        "add an attribute for size measured in cm"
    application updates json file according to comment
        makes sure still conforms to schema
        does not violate rules yet to be enumerated
    Task requires instructor library
        https://youtu.be/VllkW63LWbY?si=yrlO0Iui-ZlvQUDr
        https://www.youtube.com/watch?v=VllkW63LWbY
    Optional: use PydanticAI for Agent Based Approach

Short term
    Look at Radlex Raw
    Try to program LLM to go through batches of terms and determine which ones are imaging findings
    return big list of all imaging findings and their codes
    (eventually try also to include tags defined with them from RNSA)
        https://archive.rsna.org/2016/Specialty%20Codes.pdf
    Note: Spreadsheet located elsewhere
    Communicate in process. Show response model
    Show proposed prompt and trial outputs

OpenAI
      https://blog.gitguardian.com/how-to-handle-secrets-in-python/
    Video at the link can be rewatched to ensure proper obfuscation of the open_ai keys
Also:
    be certain to install "task" and "uv"
        use "uv sync" to install dependencies
    Look into taskfile.yml
    Learn about taskfile
    edit