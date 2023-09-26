const MsgBox = ({message, setMessage, className}: any) => {
    /*const [postMsg, setMsg] = useState("Write something awesome! :D");*/
    
    return (
        <textarea
            rows={3}
            cols={30}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            className={className}
        />
    );
};

export default MsgBox;